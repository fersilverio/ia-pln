import os.path as op
import pandas as pd


# LÊ OS DADOS DO ARQUIVO CSV
def le_dados(arquivo):
	if op.isfile(arquivo):
		dados = pd.read_csv(arquivo, encoding='utf-8')
		dados.columns = ['review', 'sentiment']
		#dados.head()
		#dados['sentiment'].value_counts()
		dados['sentiment_num'] = dados.sentiment.map({'negative':0, 'positive':1})
		# dados.head()
		return dados