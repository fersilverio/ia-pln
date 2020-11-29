from leitura import *
from processamento import *
from extracao import *

dados = le_dados('dados.csv')

base = dados['review'].apply(processa_texto)

b_train, b_test = tf_idf(base)

print(b_train)
