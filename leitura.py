# LÊ OS DADOS DO ARQUIVO CSV
def le_dados(arquivo):
	import os.path as op
	import pandas as pd

	if op.isfile(arquivo):
		dados = pd.read_csv(arquivo, converters={'sentiment': lambda x: int(x == 'positive')})
		return dados


# DIVIDE OS DADOS DE TREINO E TESTE
def divide_dados(dados):
	# DIVISÃO PARA OS DADOS DE TREINO
	train_reviews = dados.review[:40000]
	train_sentiments = dados.sentiment[:40000]

	# DIVISÃO PARA OS DADOS DE TESTE
	test_reviews = dados.review[40000:]
	test_sentiments = dados.sentiment[40000:]

	return train_reviews, train_sentiments, test_reviews, test_sentiments