# GGNN_Graph2Seq

# 包含Greed Search和Beam Search的两种Decoder实现
## Beam Search的实现
### 实现的Beam Search的思想采用的是Pointer-Network中的Beam Search思想
### Beam Search目前每次迭代时的搜索空间是 Beam Size * 2 其中 Beam Size == 10
### Beam Search目前的排序规则采用的是对序列中每个元素的log概率之和的平均值进行的排序

# 注意：
## 使用ggnn+variable+decoder_greedsearch_predict.py时需引入Hypothesis.py，Hypothesis.py是作为记录每次Beam Search之后的状态类
