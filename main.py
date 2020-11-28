'''
# 1. LER DADOS
# le_dados() # data_reading.py
# 1.1. CONVERTER ARQUIVO EM DISCO PARA OBJETO EM MEMÓRIA

# 2. PROCESSAR TEXTOS
# processa_texto() # processamento.py
#
# 2.1. DEIXAR TODAS AS PALAVRAS EM MINÚSCULO
# 2.2. REMOVER SÍMBOLOS FORA DO ALFABETO
# 2.3. TOKENIZER
# 2.4. REMOVER PALAVRAS VAZIAS (STOP WORDS)
# 2.5. NORMALIZAR (LEMMATIZING)
# 2.6. RETORNAR PARA TEXTO

# 3. REPRESENTAÇÃO TEXTUAL
# 

# 4. CLASSIFICAÇÃO
# 

# 5. MÉTRICAS
# 
'''

from leitura import *
from processamento import processa_texto

print('ANTES:\n\n')
print(dt)
base = dt['review'].apply(processa_texto)
print('DEPOIS:\n\n')
print(base)


'''
#acessando o original no dt, ou seja a review da posicao 3 do dataframe
print(dt['review'][3])
#acessando a posicao 3 de base que é o resultado do processamento de texto em cima do dt
print(base[3])
'''

'''
# O ORIGINAL A SER USADO QUANDO O TRABALHO ESTIVER PRONTO#
base = data['review'].apply(processa_texto)
print(base)
'''