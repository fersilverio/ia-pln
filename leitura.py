import os.path as op
import pandas as pd


# LÊ OS DADOS DO ARQUIVO CSV
def le_dados(arquivo):
	if op.isfile(arquivo):
		dados = pd.read_csv(arquivo, converters={'sentiment': lambda x: int(x == 'positive')})
		return dados