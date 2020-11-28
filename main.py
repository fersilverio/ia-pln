'''
1. LEITURA DE DADOS
Os dados são essenciais para as análises textuais e serão obtidos através de um arquivo com 50.000 críticas de filmes divididas igualmente em duas categorias (crítica positiva e crítica negativa).
1.1. le_dados(arquivo) - Converter arquivo CSV em disco para objeto em memória.
'''
from leitura import * ###################### remover ao terminar
# from leitura import le_dados
# dados = le_dados('input/imdb-dataset.csv')

'''
2. PROCESSAMENTO DE TEXTO
O processamento do texto é essencial para retirar "impurezas", como tags, símbolos ou palavras muito comuns que não ajudam a identificar a classe do texto.
2.1. converte_minuscula(texto) - Deixar todo o texto em minúsculo.
2.2. remove_html(texto) - Remover tags HTML.
2.3. remove_acentos(texto) - Remover acentos, cedilhas e similares.
2.4. remove_pontos(texto) - Remover pontuações que podem interferir no texto.
2.5. remove_conteudo_colchetes(texto) - Remover colchetes e todo seu texto interno.
2.6. remove_caracteres_especiais(texto) - Remover caracteres especiais que podem interferir no texto.
2.7. converte_token(texto) - Transformar o texto em um vetor de palavras.
2.8. remove_palavras_vazias(vetor) - Remover palavras vazias (stop words), que são palavras muito comuns.
2.9. normaliza(vetor) - Aproximar as palavras de seus radicais.
2.10. converte_string(vetor) - Converter vetor de palavras em uma só palavra.
'''
from processamento import processa_texto
# criticas = dados['review'].apply(processa_texto)

'''
3. EXTRAÇÃO DE CARACTERÍSTICAS
É necessário transformar o texto obtido pela etapa anterior em um vetor numérico para ser usado como entrada pelo classificador. As representações que deverão ser utilizadas são: Bag of Words e Term Frequency–Inverse Document Frequency (TF-IDF).
3.1.
3.2.
3.3.
'''
# from extracao import *
# listas = extracao(criticas)

'''
4. CLASSIFICAÇÃO
É agora que é realizada a identificação do sentimento da crítica. Alguns classificadores populares: Support Vector Machine (SVM), Árvores de Decisão, k-Nearest Neighbors (kNN), Classificador Naive Bayes.
4.1.
4.2.
4.3.
'''
# from classificacao import *

'''
5. AVALIAÇÃO DOS RESULTADOS
A métrica utilizada para avaliar os resultados gerados pelo Classificador será a F1-Score.
5.1.
5.2.
5.3.
'''
# from avaliacao import *


dados = le_dados('dados.csv')
dt = dados.head(10)
#print(dt)
#print(type(dados))

#print(dados.iloc[0][0])
#print(dados.columns)
#print(dados.head(10)) #mostra os registros de 0 a 9

#print(dados)


print('-------------------------   ANTES    -------------------------')
print(dt)
base = dt['review'].apply(processa_texto)
print('\n\n-------------------------   DEPOIS   -------------------------')
print('\n', base)

# y_t, y_p = classificacao(base['message_lem'], base['label_num'])


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


# https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments