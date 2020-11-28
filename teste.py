from leitura import le_dados
from processamento import *
from extracao import *

dados = le_dados('dados.csv')

base = dados['review'].apply(processa_texto)

print(base)
