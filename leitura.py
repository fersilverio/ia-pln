import os.path as op
import pandas as pd


# LÊ OS DADOS DO ARQUIVO CSV
def le_dados(arquivo):
	if op.isfile(arquivo):
		dados = pd.read_csv(arquivo, encoding='utf-8')
		dados.columns = ['review', 'sentiment']
		return dados


dados = le_dados('input/imdb-dataset.csv')
dt = dados.head(10)
#print(dt)
#print(type(dados))

#print(dados.iloc[0][0])
#print(dados.columns)
#print(dados.head(10)) #mostra os registros de 0 a 9

#print(dados)