from data_reading import *
from processamento import text_processing

#print('ANTES:\n\n')
#print(dt)
#base = dt['review'].apply(text_processing)
#print('DEPOIS:\n\n')
#print(base)
'''
#acessando o original no dt, ou seja a review da posicao 3 do dataframe
print(dt['review'][3])
#acessando a posicao 3 de base que é o resultado do processamento de texto em cima do dt
print(base[3])
'''

'''
# O ORIGINAL #
base = data['review'].apply(text_processing)
print(base)
'''