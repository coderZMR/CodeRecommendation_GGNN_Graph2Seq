from typing import Tuple, List, Any, Sequence
from sklearn.utils import shuffle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import time
import pickle

import json
import queue
import threading
import random
from collections import namedtuple, defaultdict
import gc


# 定义MLP层(3层)
# 很牛逼的变量命名以及存储方法
class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    # MLP层的参数以及存储方式
    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        # in_size -> hid_sizes -> out_size
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]
        network_params = {
            "weights": weights,
            "biases": biases
        }
        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    # 这里设计了两层的神经网络，因此直接调用就可以输出last_hidden
    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params['weights'], self.params['biases']):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden


# 多线程计算
class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int = 2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()


# glorot初始化
def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)

class ProgModel(object):

    # 默认参数
    # cls表示可以直接用类名调用
    # patience表示容忍度
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 500,
            'patience': 5,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 0.75,
            # 'l2_regulation':0.01,
            'momentum': 0.9,

            'embed_size': 300,
            'hidden_size': 300,
            'variable_size': 300,
            'variable_embed_size': 300,
            'softmax_size': 800,
            'num_timesteps': 5,
            'use_graph': True,

            # Decoder部分
            'decoder_hidden_size': 800,  # todo
            'decoder_num_layers': 1,  # todo

            'tie_fwd_bkwd': False,
            'task_ids': [0],
            'random_seed': 0,

            'train_file': 'chendata_CodeWithVariable-1000.json',
            'valid_file': 'chendata_CodeWithVariable-1000.json',
        }

    def __init__(self, args, training_file_count, valid_file_count):
        self.args = args
        self.training_file_count = training_file_count
        self.valid_file_count = valid_file_count

        # collect argument things
        data_dir = ''
        if args.data_dir is not None:
            data_dir = args.data_dir
        self.data_dir = data_dir

        # 设置运行记录文件（保存参数）
        # 命名格式如例子：2018-05-02-21-11-08_21156_log.json
        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = '.'
        # 模型存储路径
        self.log_file = os.path.join(log_dir, "%s_log.pickle" % self.run_id)
        # 最佳模型存储路径
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)
        self.best_model_checkpoint = "model_best-%s" % self.run_id

        # collect parameters
        # 读取默认配置
        params = self.default_params()
        # 找默认配置文件
        config_file = args.config_file
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))

        # 把args赋值给当前类的config，用来更新params
        config = args.config
        if config is not None:
            params.update(json.loads(config))

        self.params = params

        # 把现在的参数存入新的json文件
        # with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
        #    json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))

        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        # load data
        self.max_num_vertices = 0
        self.num_edge_types = 8
        self.annotation_size = 0
        self.mini_train_data = None
        self.mini_valid_data = None

        # 读取指定文件路径的训练集和验证集
        # self.train_data = self.load_data(params['train_file'], is_training_data=True)
        # self.valid_data = self.load_data(params['valid_file'], is_training_data=False)
        # print('start reading train data...')
        # self.mini_train_data = self.load_minidata(params['train_file'], is_training_data=True)
        # print('train_data : ',len(self.mini_train_data))
        # print('start reading valid data...')
        # self.mini_valid_data = self.load_minidata(params['valid_file'], is_training_data=False)
        # print('valid_data : ',len(self.mini_valid_data))

        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}

            # make model
            self.make_model()
            # make train step
            self.make_train_step()

            # Restore/initialize variables
            restore_file = args.restore
            if restore_file is not None:
                # self.initialize_model()
                self.restore_model2('L:\\Graph2Sequence\\' + restore_file)
                # self.restore_model(restore_file)
            else:
                self.initialize_model()

    # 搭建模型
    def make_model(self):
        self.placeholders['trg_label'] = tf.placeholder(tf.int64, [len(self.params['task_ids']), None],
                                                            name='trg_label')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int64, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')

        # Decoder部分
        self.placeholders['trg_emb'] = tf.placeholder(tf.float32, [None,], name='trg_emb')
        self.placeholders['trg_size'] = tf.placeholder(tf.int32, [None], name='trg_size')
        self.placeholders['trg_mask'] = tf.placeholder(tf.float32, [None], name='trg_mask')

        with tf.variable_scope("graph_model"):

            # 此处搭建具体的模型（虚拟函数）
            self.prepare_specific_graph_model()

            # this does the actual graph work:
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['orders_embed'])

        self.ops['losses'] = []

        # 定义输出层（g层）的神经网络,对我们的数据MLP设置成hidden_size
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'],
                                                                           self.params['hidden_size'], [],
                                                                           self.placeholders[
                                                                               'out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'],
                                                                                self.params['hidden_size'], [],
                                                                                self.placeholders[
                                                                                    'out_layer_dropout_keep_prob'])

                # 定义softmax层
                with tf.variable_scope("softmax"):
                    # self.weights['softmax_weights'] = tf.Variable(glorot_init([self.params['hidden_size']+ self.params['variable_embed_size'],27638]))
                    self.weights['softmax_weights'] = tf.Variable(glorot_init([self.params['softmax_size'],2592]))
                    self.weights['softmax_biases'] = tf.Variable(np.zeros([2592]).astype(np.float32))
                    print('softmax weights: ', self.weights['softmax_weights'])
                # 计算最后的g函数（输出函数）
                # 这里返回的是一个[b,softmax维度]的结果
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id],
                                                        self.weights['softmax_weights'],
                                                        self.weights['softmax_biases'])

                # diff = computed_values - self.placeholders['trg_label'][internal_id,:]
                # task_target_mask = self.placeholders['target_mask'][internal_id,:]
                # 加上小值保证非零
                # task_target_num = tf.reduce_sum(task_target_mask) + 1e-7
                # mask out unused values
                # diff = diff * task_target_mask

                groundtruth = self.placeholders['trg_label'][internal_id, :]    # TODO

                # reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.params['l2_regulation']), tf.trainable_variables())

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=computed_values, labels=groundtruth)
                # Decoder部分
                # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练
                # label_weights = tf.sequence_mask(
                #     self.placeholders['trg_size'], dtype=tf.float32)
                # label_weights = tf.reshape(label_weights, [-1])
                label_weights = self.placeholders['trg_mask']
                task_loss = tf.reduce_sum(loss * label_weights)
                # cost_per_token = cost / tf.reduce_sum(label_weights)

                # task_loss = tf.reduce_sum(loss)

                # correct_prediction = tf.equal(tf.argmax(computed_values, 1), groundtruth)
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                self.ops['accuracy_task%i' % task_id] = task_loss

                # 计算损失函数，平方根和
                # task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num

                # normalise loss to account for fewer task-specific example in batch,如果没有就是1.0
                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
                self.ops['losses'].append(task_loss)
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

        self.saver = tf.train.Saver()

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.freeze_graph_model:
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars

        # 学习率自动衰减
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(self.params['learning_rate'], batch, 2000, 1.0, staircase=False)

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.params['momentum'], use_nesterov=True)

        # optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        # optimizer = tf.train.AdagradOptimizer(self.params['learning_rate'])

        # 找出全部的梯度

        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []

        # 进行梯度削减
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads, global_step=batch)
        # self.ops['train_step'] = optimizer.apply_gradients(grads_and_vars)
        # Initialize newly-introduced variables
        self.sess.run(tf.local_variables_initializer())

    # 门回归
    def gated_regression(self, last_h, regreesion_gate, regression_transform, softmax_weights, softmax_biases):
        raise Exception("Models have to implement gated_regression!")

    # 具体模型
    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    # 读取数据集
    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        # 判断是否数据需要缩减
        restrict = self.args.restrict
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # 从原始数据中抽取信息
        # Get some common data out
        num_fwd_edge_types = 0
        for g in data:
            # 找出数据集中节点总数
            # 从点边点组合中的两部分点找
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            # 找出前向边的种类
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        # 找出边的种类，如果是有向图需要x2，而无向图x1
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))

        # 标注的维度
        # self.annotation_size = max(self.annotation_size, len(data[0]['node_features'][0]))

        # if is_training_data:
        #    self.raw_train_data = data
        # else:
        #    self.raw_valid_data = data

        # 把原始数据进行了处理
        return self.process_raw_graphs(data, is_training_data)
        # return data

    def load_minidata(self, filename, is_training_data: bool):
        # 读取图数据信息
        with open(filename, 'r') as f:
            data = json.load(f)
            # Get some common data out
        num_fwd_edge_types = 0
        for g in data:
            # 找出数据集中节点总数
            # 从点边点组合中的两部分点找
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            # 找出前向边的种类
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        # 找出边的种类，如果是有向图需要x2，而无向图x1
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        # self.annotation_size = max(self.annotation_size, len(data[0]['node_features'][0]))
        # print('数据边种类',num_fwd_edge_types)
        # if is_training_data:
        #    self.raw_mini_train_data = data
        # else:
        #    self.raw_mini_valid_data = data
        # print(str(filename) + ' : ' + str(len(data)))
        # print('num_edge_types: ',self.num_edge_types)

        return self.process_raw_graphs(data, is_training_data)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

        # 处理原始数据

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def run_epoch(self, epoch_name: str, data, is_training: bool):
        # chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
        #                                0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
        #                                0.037467458])

        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        read_data_time = 0
        total = 0
        processed_graphs = 0
        count = 0
        file_count = 0
        index = 1
        if is_training:
            file_count = self.training_file_count
            # index = 15
        else:
            file_count = self.valid_file_count
            index = 17
        while count < file_count:
            tempGraph = 0
            tempAcc = []
            full_path = "L:\\Graph2Sequence\\" + str(
                count + index) + ".json"
            # t = time.time()
            filestr = None
            if is_training:
                filestr = "training"
            else:
                filestr = "valid"
            t = time.time()
            data = self.load_minidata(full_path, is_training_data=is_training)
            read_data_time = time.time() - t + read_data_time
            # print(time.time()-t)
            count = count + 1
            # t = time.time()
            batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
            # print(time.time()-t)
            for step, batch_data in enumerate(batch_iterator):
                total = total + 1
                num_graphs = batch_data[self.placeholders['num_graphs']]
                # 记录已处理graphs数
                processed_graphs += num_graphs
                tempGraph += num_graphs
                if is_training:
                    batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                        'out_layer_dropout_keep_prob']
                    fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
                else:
                    batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                    fetch_list = [self.ops['loss'], accuracy_ops]

                # print(batch_data)

                result = self.sess.run(fetch_list, feed_dict=batch_data)
                (batch_loss, batch_accuracies) = (result[0], result[1])
                loss += batch_loss * num_graphs
                accuracies.append(np.array(batch_accuracies) * num_graphs)
                tempAcc.append(np.array(batch_accuracies) * num_graphs)

                print("Running %s, %s file %i, batch %i (has %i graphs). Loss so far: %.4f" % (
                epoch_name, filestr, count, total,
                num_graphs, loss / processed_graphs), end='\r')
            # data = None
            # batch_iterator = None
            del data
            del batch_iterator
            gc.collect()
            # print(tempGraph, " " ,np.sum(tempAcc, axis=0) / tempGraph)
        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        # error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        end_time = time.time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        m, s = divmod(read_data_time, 60)
        h, m = divmod(m, 60)
        read_data_time_str = "%02d:%02d:%02d" % (h, m, s)
        instance_per_sec = processed_graphs / (time.time() - start_time - read_data_time)
        return loss, accuracies, instance_per_sec, total, processed_graphs, time_str, read_data_time_str

    def train(self):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.args.restore is not None:
                valid_loss, valid_accs, valid_speed, valid_batch, valid_total_graph, valid_time, valid_read_data_time = self.run_epoch(
                    "Resumed (validation)", self.mini_valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (float("-inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_accs, train_speed, train_batch, train_total_graph, train_time, train_read_data_time = self.run_epoch(
                    "epoch %i (training)" % epoch,
                    self.mini_train_data, True)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                # errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print(
                    "\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f | train_batch: %i | train_total_graph: %i | train_time: %s | train_read_data_time: %s" % (
                    train_loss,
                    accs_str,
                    train_speed, train_batch, train_total_graph, train_time, train_read_data_time))
                valid_loss, valid_accs, valid_speed, valid_batch, valid_total_graph, valid_time, valid_read_data_time = self.run_epoch(
                    "epoch %i (validation)" % epoch,
                    self.mini_valid_data, False)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                # errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print(
                    "\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f | valid_batch: %i | valid_total_graph: %i | valid_time: %s | valid_read_data_time: %s" % (
                    valid_loss,
                    accs_str,
                    valid_speed, valid_batch, valid_total_graph, valid_time, valid_read_data_time))

                epoch_time = time.time() - total_time_start
                #                 log_entry = {
                #                     'epoch': epoch,
                #                     'time': epoch_time,
                #                     'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                #                     'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
                #                 }
                #                 log_to_save.append(log_entry)
                #                 with open(self.log_file, 'w') as f:
                #                     json.dump(log_to_save, f, indent=4)

                val_acc = np.sum(valid_accs)  # type: float
                if val_acc > best_val_acc:
                    # self.save_model(self.best_model_file)
                    self.save_model2('L:\\Graph2Sequence\\')
                    print("  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')" % (
                    val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params[
                        'patience'])
                    break

    # 初始化模型参数
    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    # 保存模型，将模型中的变量和参数存入pickle文件
    # 这里倒蛮牛逼的
    def save_model(self, path: str) -> None:
        weights_to_save = {}
        # 利用get_collection获得graph中的变量
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "params": self.params,
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def save_model2(self, path: str) -> None:
        self.saver.save(self.sess, path + self.best_model_checkpoint)

    # 读取模型，将模型的变量和参数取出，将每个变量赋上存储的值
    # 如果出现没有对上号的变量，则随机初始化
    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    # 将读取到的weights赋值给对应的variable
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print("Freshly initializing %s since no saved value was found." % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print("Saved weights for %s not used by model." % var_name)

            # 将所有要初始化的变量合并到一起
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)

    def restore_model2(self, path: str) -> None:
        print("Restore...")
        # self.sess = tf.Session()
        self.saver.restore(self.sess, path)
        print("Restore done!")


# 将graph转换成[num_edge_types, max_n_vertices, max_n_vertices]邻接矩阵
def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=False):
    # print(max_n_vertices,num_edge_types)
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
    # 序号的起始是1，因此都记得要减1
    for src, e, dest in graph:
        # 如果graph是通过补0扩展得到的，那么补充的部分应该忽略
        if (e == 0 and src == 0 and dest == 0):
            continue
        amat[e - 1, dest - 1, src - 1] = 1
        amat[e - 1 + bwd_edge_offset, src - 1, dest - 1] = 1
    return amat


class DenseGGNNProgModel(ProgModel):
    def __init__(self, args, training_file_count, valid_file_count):
        super().__init__(args, training_file_count, valid_file_count)
        # 取得embedding层预训练参数
        # self.embedding_weight = self.load_embedding_weight()

        # 创建变量名模型所需要的词表
        # self.Variable(dtype=tf.float32, [])

    # 读取treelstm模型中的embedding层
    def load_embedding_weight(self):
        load = np.load('data/treelstm.npy', encoding="latin1").item()
        return load['word_embedding/weights']

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 1,
            'graph_state_dropout_keep_prob': 0.75,
            'task_sample_ratios': {},
            'use_edge_bias': True,
        })
        return params

    # 实现抽象方法
    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        e_dim = self.params['embed_size']
        v_dim = self.params['variable_size']

        # input
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None,
                                                                            name='edge_weight_dropout_keep_prob')

        self.placeholders['input_orders'] = tf.placeholder(tf.int32, [None, None, ], name='input_orders')
        self.weights['index2vector'] = tf.Variable(dtype=tf.float32,
                                                   initial_value=np.random.uniform(-0.5, 0.5, [2494, e_dim]))

        self.placeholders['output_orders'] = tf.placeholder(tf.int32, [None, None, ], name='output_orders')
        self.weights['output2vector'] = tf.Variable(dtype=tf.float32,
                                                   initial_value=np.random.uniform(-0.5, 0.5, [2592, e_dim]))

        # 这里加入变量名模型
        self.placeholders['variable_orders'] = tf.placeholder(tf.int32, [None, 10, ], name='variable_order')
        self.weights['variable_index2vector'] = tf.Variable(dtype=tf.float32,
                                                            initial_value=np.random.uniform(-0.5, 0.5, [2245, v_dim]))

        # 对变量名和api进行embeddding,变量名维度也是[b, v, v_dim]
        self.placeholders['variable_orders_embed_first'] = tf.nn.embedding_lookup(self.weights['variable_index2vector'],
                                                                                  self.placeholders['variable_orders'])
        self.placeholders['variable_orders_embed'] = tf.layers.dense(self.placeholders['variable_orders_embed_first'],
                                                                     self.params['variable_embed_size'], tf.tanh)
        # self.placeholders['variable_orders_embed'] = tf.nn.dropout(self.placeholders['variable_orders_embed'],self.placeholders['out_layer_dropout_keep_prob'])
        self.placeholders['variable_orders_embed'] = tf.layers.dense(self.placeholders['variable_orders_embed'],
                                                                     self.params['variable_embed_size'], tf.tanh)
        # self.placeholders['variable_orders_embed'] = tf.nn.dropout(self.placeholders['variable_orders_embed'],self.placeholders['out_layer_dropout_keep_prob'])
        self.placeholders['variable_orders_embed'] = tf.layers.dense(self.placeholders['variable_orders_embed'],
                                                                     self.params['variable_embed_size'], tf.tanh)
        # self.placeholders['variable_orders_embed'] = tf.nn.dropout(self.placeholders['variable_orders_embed'],self.placeholders['out_layer_dropout_keep_prob'])

        # attention
        # self.variable_vectors = tf.contrib.seq2seq.LuongAttention(
        #    self.params['variable_embed_size'], self.placeholders['variable_orders_embed'],
        #    memory_sequence_length=[10], scale=True
        # )

        self.placeholders['orders_embed'] = tf.nn.embedding_lookup(self.weights['index2vector'],
                                                                   self.placeholders['input_orders'])
        # Decoder部分
        self.placeholders['trg_emb'] = tf.nn.embedding_lookup(self.weights['output2vector'],
                                                                   self.placeholders['output_orders'])


        # self.placeholders['orders_embed_first'] = tf.nn.embedding_lookup(self.weights['index2vector'],self.placeholders['input_orders'])
        # self.placeholders['orders_embed'] = tf.layers.dense(self.placeholders['orders_embed_first'], h_dim,tf.tanh)

        # 利用max_pooling选出适合的特征
        # shape = self.placeholders['variable_orders_embed'].get_shape().as_list()

        # 使用mask过滤掉0的部分
        self.placeholders['variable_mask'] = tf.placeholder(tf.float32, [None, 10, self.params['variable_embed_size']],
                                                            name='variable_mask')
        self.placeholders['variable_orders_embed'] = self.placeholders['variable_orders_embed'] * self.placeholders[
            'variable_mask']

        # self.placeholders['variable_orders_embed'] = self.Position_Embedding_Attention(self.placeholders['variable_orders_embed'],self.params['variable_embed_size'])
        self.variable_embeddings = self.placeholders['variable_orders_embed']
        # self.variable_embeddings = tf.expand_dims(self.placeholders['variable_orders_embed'],axis=len(shape))
        print('self.variable_embedding ', self.variable_embeddings.get_shape().as_list())
        # print('variable_embeddings shape: ',self.variable_embeddings.get_shape().as_list())
        # self.variable_vectors = tf.nn.max_pool(self.variable_embeddings,[1,shape[1],1,1],[1,1,1,1],'VALID')
        # print('variable after max_pooling: ',self.variable_vectors.get_shape().as_list())
        # self.variable_vectors = tf.squeeze(self.variable_vectors,[1,3]) # [b, v_dim]

        # print('final vairable vector: ',self.variable_vectors)
        # self.placeholders['att'] = tf.placeholder(tf.int32, [None, 1,], name='att')
        # self.query = tf.nn.embedding_lookup(self.weights['index2vector'],self.placeholders['att'])
        # print('self.query ', self.query.get_shape().as_list())
        # [b, v, h_dim]
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32,
                                                                          [None, None, self.params['hidden_size']],
                                                                          name='node_features')
        # [b, v]
        self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None, self.params['hidden_size']],
                                                        name='node_mask')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, ())
        # [b,e,v,v]
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,
                                                               [None, self.num_edge_types, None, None])
        # [b,e,v,v] -> [e,b,v,v]
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])

        # weights
        # 边的weight和biase
        # weight -> [num_edge_types,dim,dim],初始化graph的weight
        # biase -> [num_edge_types,1,dim]，初始化graph的biase
        # 现在修改weights成[edge_type, h_dim, h_dim]
        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, h_dim, h_dim]),
                                                   name='edge_weights')
        if self.params['use_edge_bias']:
            self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32),
                                                      name='edge_biases')
        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(h_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell

    # 实现抽象方法
    # 这里定义了每次信息传递过程中需要的计算
    # 包括了边的传播和GRU单元的计算
    def compute_final_node_representations(self) -> tf.Tensor:
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        e_dim = self.params['embed_size']
        # [b, v, h] -> [b*v, h]
        # 输入[batchsize, words] -> [batchsize, words, vec]
        orders = self.placeholders['orders_embed']
        # self.orders_embed = tf.nn.embedding_lookup(self.weights['index2vector'],orders)

        # orders_embed: [b, v, h_dim] -> [b*v, h_dim]
        orders_embed = tf.reshape(orders, [-1, h_dim])
        # h = self.placeholders['initial_node_representation']
        # h = tf.reshape(h, [-1, h_dim])

        # precompute edge biases
        # 计算出每种edge的biase
        if self.params['use_edge_bias']:
            biases = []
            # unstack成(e; t) -> 最终变成[b*v, h] , t:[b, v, v]
            for edge_type, a in enumerate(tf.unstack(self.__adjacency_matrix, axis=0)):
                # 变形成[b*v, 1]
                summed_a = tf.reshape(tf.reduce_sum(a, axis=-1), [-1, 1])
                # 相乘后[b*v, h]
                # self.weights['edge_biases'][edge_type] --- [1, h_dim]
                biases.append(tf.matmul(summed_a, self.weights['edge_biases'][edge_type]))

        # GRU单元信息传递
        with tf.variable_scope("gru_scope") as scope:
            # 每个节点进行一次信息传递（和每种边进行矩阵乘法），然后放入GRU中利用门结构进行信息提取
            for i in range(self.params['num_timesteps']):
                # 共享每一步t的gru参数
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                # 遍历每一种edge_type
                for edge_type in range(self.num_edge_types):
                    # [b*v, e] * [e_dim, h_dim] -> [b*v, h_dim]
                    m = tf.matmul(orders_embed, tf.nn.dropout(self.weights['edge_weights'][edge_type],
                                                              self.placeholders['edge_weight_dropout_keep_prob']))
                    if self.params['use_edge_bias']:
                        m += biases[edge_type]
                    # [b*v, h_dim] -> [b, v, h_dim]
                    m = tf.reshape(m, [-1, v, h_dim])
                    # 这里存储act内容，因为是遍历每个edgetype，因此把每个信息都加到一起
                    if edge_type == 0:
                        # __adjacency_matrix[edge_type] -> [b, v, v]
                        # [b, v, v] * [b, v, dim] -> [b, v, dim]
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])

                # 向gru单元里输入取出hidden,[b*v, h_dim]
                orders_embed = self.weights['node_gru'](acts, orders_embed)[1]
            # [b, v, h_dim]
            last_h = tf.reshape(orders_embed, [-1, v, h_dim])
        return last_h

    # 计算模型的输出部分
    def gated_regression(self, last_h, regression_gate, regression_transform, softmax_weights, softmax_biases):
        # last_h: [b, v, h]
        # gate_input: [b, v, 2*h] -> [b*v, 2*h]
        # 把最后一次time step的GRU结果拿来和initial_node_representation合并
        # gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=2)
        gate_input = tf.concat([last_h, self.placeholders['orders_embed']], axis=2)
        gate_input = tf.reshape(gate_input, [-1, 2 * self.params["hidden_size"]])

        # [b*v, h]
        last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])

        # regression_gate: [b*v, 2*h] -> [b*v, 1] (NEW! 现在修改成[b*v, 2*h]->[b*v, h])
        # regression_transform: [b*v, h] -> [b*v, 1] (NEW! 现在修改成[b*v, h]->[b*v, h])
        # gated_outputs: [b*v, 1] (NEW! 现在修改成[b*v, h])
        # 由regreesion_gate充当了一个门结构，决定last_h中有多少传递进来（有点像lstm的遗忘门）
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)

        # [b*v, h] -> [b, v, h]
        gated_outputs = tf.reshape(gated_outputs, [-1, self.placeholders['num_vertices'], self.params["hidden_size"]])

        # 用node_mask确定哪些点需要mask掉
        masked_gated_outputs = gated_outputs * self.placeholders['node_mask']
        # self.variable_embeddings = tf.squeeze(self.variable_embeddings,axis=-1)
        # self.variable_vectors = tf.reduce_sum(self.variable_embeddings, axis = 1)
        # 尝试性操作，把所有的节点加到一起，建立[b,h]维度
        try_sum = tf.reduce_sum(masked_gated_outputs, axis=1)
        print('graph vector: ', try_sum)
        self.variable_vectors = self.attention(self.variable_embeddings, try_sum, self.params["variable_size"])
        print('variable vector: ', self.variable_vectors)
        # text_shape = self.variable_vectors.get_shape()

        # zeros = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0, 0,[256,300]))
        text_vector = self.variable_vectors
        text_vector = tf.reshape(text_vector, [-1, 1, 300])
        print('variable vector reshape: ', text_vector)
        # code_shape = try_sum.get_shape()
        code_vector = tf.reshape(try_sum, [-1, 1, 300])
        print('graph vector reshape: ', code_vector)
        union_inputs = tf.concat([code_vector, text_vector], axis=1)
        print('union_inputs: ', union_inputs)
        union_query = tf.concat([text_vector, code_vector], axis=1)
        print('union_query: ', union_query)
        # query = tf.nn.embedding_lookup(self.weights['index2vector'],self.placeholders['att'])
        # print('query: ',query)
        # union_query = tf.concat([try_sum,self.variable_vectors],axis=1)

        # 合并code和变量名的特征，300+100维
        inte_vectors = self.attention2(union_inputs, union_query, 300)
        inte_vectors = tf.reshape(inte_vectors, [-1, 600])
        print('inte_vectors: ', inte_vectors)
        inte_vectors = tf.layers.dense(inte_vectors, self.params['softmax_size'], tf.tanh)
        print('after concat: ', inte_vectors.get_shape().as_list())
        # 计算和: [b, v] -> [b]
        # output = tf.reduce_sum(masked_gated_outputs, axis = 1)

        # 不加Decoder部分，预测单个API
        # 对抽取的特征做softmax分类
        # output = tf.matmul(inte_vectors, self.weights['softmax_weights']) + self.weights['softmax_biases']
        # self.output = output
        # return output

        # inte_vectors为encoder的输出
        # Decoder部分
        # 使用dyanmic_rnn构造解码器。
        # 输出的dec_outputs为每一步顶层GRU的输出。dec_outputs的维度是 [batch_size, max_time, HIDDEN_SIZE]。
        # self.placeholders['trg_emb']是API的embedding
        # self.placeholders['trg_size']是目标API序列的长度
        # 设置GRU单元大小及Decoder的层数
        HIDDEN_SIZE = self.params['decoder_hidden_size']
        NUM_LAYERS = self.params['decoder_num_layers']
        # 定义Decoder所使用的GRU结构
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)]
        )
        with tf.variable_scope("decoder"):
            dec_outputs, _ = tf.nn.dynamic_rnn(
                self.dec_cell, self.placeholders['trg_emb'], sequence_length=self.placeholders['trg_size'],
                initial_state=tuple([inte_vectors[:, :] for _ in range(NUM_LAYERS)]))
        # 计算解码器每一步的log perplexity。这一步与语言模型代码相同
        output = tf.reshape(dec_outputs, [-1, self.params['decoder_hidden_size']])
        logits = tf.matmul(output, softmax_weights) + softmax_biases
        return logits

    # 数据预处理以及minibatch
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool, bucket_sizes=None) -> Any:
        # 生成不同的bucket，装入和它大小相近的数据进行分组，之后填充零
        if bucket_sizes is None:
            bucket_sizes = np.array(list(range(1, 6200, 2)))
        bucketed = defaultdict(list)
        # 获取数据中node_features的维度（默认的分子数据中是5）
        # x_dim = len(raw_data[0]["node_features"][0])

        # 遍历每个数据
        # print(len(raw_data))
        for d in raw_data:
            graph = d['graph']
            orders = d['orders'][0]
            # att = 35023
            # for g in graph:
            #    if g[1] == 4:
            #        att = orders[g[0]-1]
            #        break
            # 返回第一个满足判断的index
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                              for v in [e[0], e[2]]]))
            # print(chosen_bucket_idx)
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
            # 活跃节点数（或相当于节点数）
            n_active_nodes = len(d["orders"][0])
            # print(n_active_nodes)
            # print(d["orders"])

            # 添加变量名mask,之前数据集合中有-1，现在转换成unk
            cur_variable_indexes = []
            variables = d['variable']
            if len(variables) > 10:
                variables = variables[0:10]
                # variables = variables[len(variables) - 10:len(variables)]
            for idx, i in enumerate(variables):
                if i == -1:
                    cur_variable_indexes.append(2244)
                else:
                    cur_variable_indexes.append(i)
                # print(len(cur_variable_indexes))
            # 将节点数相近的并在一起（用稍大一点的chosen_bucket_size约束），同时添加mask用作屏蔽
            # 比如：原始数据有9个节点，我们设置的chosen_bucket_size为10，那么会添加一个纯0的节点，
            # 并且将增加的这部分也反映到邻接矩阵上，最后添加到mask中，让它的mask为0

            # 如果这里引入了embedding，需要将加入的init的vector对应到字典的unknown

            # 'init': d["node_features"] + [[0 for _ in range(x_dim)] for _ in range(chosen_bucket_size - n_active_nodes)],
            bucketed[chosen_bucket_idx].append({
                'adj_mat': graph_to_adj_mat(d['graph'], chosen_bucket_size, self.num_edge_types,
                                            self.params['tie_fwd_bkwd']),
                'orders': [d["orders"][0] + [0 for _ in range(chosen_bucket_size - n_active_nodes)]],
                'labels': d["targets"][0],
                'mask': [[1.0 for _ in range(self.params['hidden_size'])] for _ in range(n_active_nodes)] +
                        [[0. for _ in range(self.params['hidden_size'])] for _ in
                         range(chosen_bucket_size - n_active_nodes)],
                'variable': [cur_variable_indexes]
                # 'att': [att]
            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                # 打乱每个bucket（包含了相同节点）的数据
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    # 抽样？（似乎是将一部分数据的labels变成None）
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        # 用bucket_idx表示每个bucket中执行的次数，比如[[1,1,1,1],[2,2,2],[3,3,3,3,3,3]]
        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'] + 1)]
                          for bucket_idx, bucket_data in bucketed.items()]

        # 铺平成[1,1,1,1,2,2,2,3,3,3,3,3,3]
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return (bucketed, bucket_sizes, bucket_at_step)

    def make_batch(self, elements):
        batch_data = {'adj_mat': [], 'orders': [], 'output_orders':[], 'labels': [], 'node_mask': [],'tar_size':[],'task_masks': [], 'variables': [],
                      'variable_masks': []}
        # 逐数据将信息填入
        max_variable_size = 10
        target_task_values = []
        # print('cur batch variable length: ',max_variable_size)
        for d in elements:
            variable_length = len(d['variable'][0])
            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['orders'].append(d['orders'])
            batch_data['node_mask'].append(d['mask'])
            # print(d['variable'])
            variables = [idx for idx in d['variable'][0][:10]] + [2244 for _ in
                                                                  range(max_variable_size - variable_length)]
            # print(np.array(variables).shape)
            batch_data['variables'].append(variables)


            # Decoder部分
            #for target_val in d['labels']:
            target_val_array = d['labels']

           # target_task_values.append(target_val_array)
            # print(target_val_array)
            target_length = len(target_val_array)
            if target_length >= 60:
                target_val_array = target_val_array[0:59]
                target_val_array.append(2590)
            else:
                target_val_array.append(2590)
            target_length = len(target_val_array)
            output_orders = []
            output_orders.append(2589)
            for i in range(60):
                if i < target_length:
                    target_task_values.append(target_val_array[i])
                else:
                    target_task_values.append(2591)
            for i in range(59):
                if i < target_length:
                    output_orders.append(target_val_array[i])
                else:
                    output_orders.append(2591)

            variable_mask = []
            for i in range(max_variable_size):
                if i < variable_length:
                    variable_mask.append([1 for _ in range(self.params['variable_embed_size'])])
                else:
                    variable_mask.append([0 for _ in range(self.params['variable_embed_size'])])

            for i in range(target_length):
                batch_data['task_masks'].append(1)
            for i in range(60 - target_length):
                batch_data['task_masks'].append(0)

            batch_data['output_orders'].append(output_orders)
            batch_data['tar_size'].append(target_length)
            #batch_data['task_masks'].append(target_task_mask)
            batch_data['variable_masks'].append(variable_mask)
            # batch_data['att'].append(d['att'])
        batch_data['labels'].append(target_task_values)
        return batch_data

    def make_minibatch_iterator(self, data, is_training: bool):
        # print(data[1])
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            # 打乱需要执行的顺序
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                # 打乱指定bucketed中同长度的数据
                np.random.shuffle(bucketed_data)

        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        # 获得batchsize个数据
        for step in range(len(bucket_at_step)):
            # 选中一个bucket，用于寻找该数量nodes的集合
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            # 将batchsize长个数据做成batch
            batch_data = self.make_batch(elements)

            num_graphs = len(batch_data['orders'])
            # initial_representations = batch_data['init']

            # 补零，将节点的维度拉伸到hidden_size
            # initial_representations = self.pad_annotations(initial_representations)
            # print(initial_representations)

            batch_data['orders'] = np.squeeze(batch_data['orders'], axis=1)
            if len(batch_data['orders'].shape) == 1:
                batch_data['orders'] = np.expand_dims(batch_data['orders'], axis=0)

            if len(batch_data['labels']) == 0:
                continue
            # 打印order
            # print(batch_data['orders'])
            # self.placeholders['initial_node_representation']: initial_representations,

            # print('################################################################################')
            # print('order: ',batch_data['orders'].shape)
            # print('target: ',np.array(batch_data['labels']).shape)
            # print('target_mask: ',np.array(batch_data['task_masks']).shape)
            # print('num_vertices: ',bucket_sizes[bucket])
            # print('adjacency_matrix: ',np.array(batch_data['adj_mat']).shape)
            # print('variable_orders: ',np.array(batch_data['variables']).shape)
            # print('variable_mask: ',np.array(batch_data['variable_masks']).shape)

           # print('################################################################################')
           # print(batch_data['task_masks'])
            batch_feed_dict = {
                self.placeholders['input_orders']: batch_data['orders'],  # 顺序
                self.placeholders['output_orders']: batch_data['output_orders'],
                # self.placeholders['trg_label']: np.transpose(batch_data['labels'], axes=[1, 0]),
                # self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                self.placeholders['trg_label']: batch_data['labels'],
                #self.placeholders['target_mask']: batch_data['task_masks'],
                self.placeholders['num_graphs']: num_graphs,  # 多少个数据
                self.placeholders['num_vertices']: bucket_sizes[bucket],  # 点数量（补0后）
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],  # 邻接矩阵
                self.placeholders['node_mask']: batch_data['node_mask'],  # 节点mask
                self.placeholders['variable_orders']: batch_data['variables'],  # 变量
                self.placeholders['variable_mask']: batch_data['variable_masks'],  # 变量mask
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: dropout_keep_prob,
                # self.placeholders['att']:batch_data['att']
                # Decoder部分
                self.placeholders['trg_size']: batch_data['tar_size'],
                self.placeholders['trg_mask']: batch_data['task_masks']
            }

            bucket_counters[bucket] += 1

            yield batch_feed_dict

    # 补零 [b, v, annotation_size] -> [b, v, h_dim]
    def pad_annotations(self, annotations):
        return np.pad(annotations, pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.annotation_size]],
                      mode='constant')

    def evaluate_one_batch(self, initial_node_representations, orders, adjacency_matrices, node_masks=None):
        num_vertices = len(initial_node_representations[0])
        if node_masks is None:
            node_masks = []
            for r in initial_node_representations:
                node_masks.append([1. for _ in r] + [0. for _ in range(num_vertices - len(r))])

        # self.placeholders['initial_node_representation']: self.pad_annotations(initial_node_representations),
        batch_feed_dict = {
            self.placeholders['input_orders']: orders,
            self.placeholders['num_graphs']: len(initial_node_representations),
            self.placeholders['num_vertices']: len(initial_node_representations[0]),
            self.placeholders['adjacency_matrix']: adjacency_matrices,
            self.placeholders['node_mask']: node_masks,
            self.placeholders['graph_state_keep_prob']: 1.0,
            self.placeholders['out_layer_dropout_keep_prob']: 1.0,
        }

        fetch_list = self.output
        result = self.sess.run(fetch_list, feed_dict=batch_feed_dict)
        return result

    # 验证集测试
    def example_evaluation(self):

        n_example_molecules = 10
        with open('molecules_valid.json', 'r') as valid_file:
            example_molecules = json.load(valid_file)[:n_example_molecules]
        example_molecules, _, _ = self.process_raw_graphs(example_molecules, is_training_data=False,
                                                          bucket_sizes=np.array([29]))
        batch_data = self.make_batch(example_molecules[0])
        print(self.evaluate_one_batch(batch_data['init'], batch_data['orders'], batch_data['adj_mat']))

    def attention2(self, inputs, query, attention_size):

        # q = tf.expand_dims(query,1)
        q = query
        print("query: ", q.get_shape().as_list())
        # inputs:[None,10,300]表示变量名
        # query:[None,10,300]表示图
        hidden_size = attention_size

        # Trainable parameters
        # w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # u_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # v_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        # w_omega = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0.05, 0.05,[hidden_size, attention_size]))
        # u_omega = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0.05, 0.05,[hidden_size, attention_size]))
        # v_omega = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0.05, 0.05,[hidden_size]))
        w_omega = tf.Variable(glorot_init([hidden_size, attention_size]))
        u_omega = tf.Variable(glorot_init([hidden_size, attention_size]))
        v_omega = tf.Variable(dtype=tf.float32, initial_value=np.random.uniform(-0.05, 0.05, [hidden_size]))

        with tf.name_scope('v'):
            # print('inputs: ', inputs)
            # print('w_omega: ',w_omega)
            # print('inputs: ', query)
            # print('w_omega: ',u_omega)
            # v = tf.tanh(tf.tensordot(w_omega,inputs, axes=1) + tf.tensordot(u_omega, query, axes=1))
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + tf.tensordot(q, u_omega, axes=1))

        # print('inputs: ', tf.tensordot(inputs, w_omega, axes=1))
        vu = tf.tensordot(v, v_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')

        # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1),1)
        output = inputs * tf.expand_dims(alphas, -1)

        return output

    def attention(self, inputs, query, attention_size):

        q = tf.expand_dims(query, 1)
        q = tf.tile(q, [1, 10, 1])
        # inputs:[None,10,300]表示变量名
        # query:[None,10,300]表示图
        hidden_size = attention_size

        # Trainable parameters
        # w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # u_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # v_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        # w_omega = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0.05, 0.05,[hidden_size, attention_size]))
        # u_omega = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0.05, 0.05,[hidden_size, attention_size]))
        # v_omega = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0.05, 0.05,[hidden_size]))
        w_omega = tf.Variable(glorot_init([hidden_size, attention_size]))
        u_omega = tf.Variable(glorot_init([hidden_size, attention_size]))
        v_omega = tf.Variable(dtype=tf.float32, initial_value=np.random.uniform(-0.05, 0.05, [hidden_size]))

        with tf.name_scope('v'):
            # print('inputs: ', inputs)
            # print('w_omega: ',w_omega)
            # print('inputs: ', query)
            # print('w_omega: ',u_omega)
            # v = tf.tanh(tf.tensordot(w_omega,inputs, axes=1) + tf.tensordot(u_omega, query, axes=1))
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + tf.tensordot(q, u_omega, axes=1))

        # print('inputs: ', tf.tensordot(inputs, w_omega, axes=1))
        vu = tf.tensordot(v, v_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')

        # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1),1)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        return output

    def softmax(self, array):
        now_array = np.array(array)
        exp_sum = 0.0
        new_array = []
        for num in now_array[0]:
            exp_sum += np.exp(num)
        for num in now_array[0]:
            new_array.append(np.exp(num) / exp_sum)
        return np.array(new_array)

Parser = namedtuple('parser',['data_dir','config_file','config','restore','restrict','freeze_graph_model'])
parser = Parser(data_dir='',config_file=None,config=None,restore=None,restrict=None,freeze_graph_model=False)
model = DenseGGNNProgModel(parser,training_file_count=6,valid_file_count=6)

evaluation = False
if evaluation:
    model.example_evaluation()
else:
    model.train()
