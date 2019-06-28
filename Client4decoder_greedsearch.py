import zerorpc
import sys

# graphRepresent = sys.argv[1]
# graphVocab = sys.argv[2]
# variableName = sys.argv[3]

c = zerorpc.Client()
c.connect("tcp://127.0.0.1:6666")

graphRepresent = "[[1,1,2],[2,1,3],[2,2,4],[2,2,5],[3,1,6],[3,1,7],[3,1,8],[3,4,9],[7,1,10],[8,1,11],[10,3,12],[11,1,5],[12,1,4]]"
graphVocab = "{1:'java.lang.String.getBytes()',2:'java.lang.String.Null',3:'if',4:'java.lang.String.Constant',5:'java.lang.String.Constant',6:'condition',7:'then',8:'else',9:'hole',10:'java.security.PrivateKey.Constant',11:'java.security.PrivateKey.getAlgorithm()',12:'java.security.PrivateKey.getAlgorithm()'}"
variableName = "sign message digest algorithm pk byte mode encryption"
ans = c.predict(graphRepresent,graphVocab,variableName)
print('startrecord')
print(ans)