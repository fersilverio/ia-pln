import numpy as np
import nltk
from sklearn.preprocessing import LabelBinarizer
import string,unicodedata
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
import os
import warnings
warnings.filterwarnings('ignore')


from processamento import *
from leitura import *
from extracao import *
from classificacao import *

#Leitura do arquivo de dados

imdb_data = le_dados('dados.csv')

#Divisão dos dados entre treino e teste

#Divisão para os dados de treino
train_reviews = imdb_data.review[:40000]
train_sentiments = imdb_data.sentiment[:40000]
#Divisão para os dados de teste
test_reviews = imdb_data.review[40000:]
test_sentiments = imdb_data.sentiment[40000:]

#Aplicando o pré-processamento

imdb_data['review'] = imdb_data['review'].apply(processa_texto)

#Reviews para o treino pré processadas
processed_train_reviews = imdb_data.review[:40000] 

#Reviews para o teste pré processadas
processed_test_reviews = imdb_data.review[40000:] 

#Transformando em matrizes tf_idf para treino e teste
tf_idf_train_reviews, tf_idf_test_reviews = tf_idf(processed_train_reviews, processed_test_reviews)

#Transformando em matrizes BOW para treino e teste
bow_train_reviews, bow_test_reviews = bow(processed_train_reviews, processed_test_reviews) #ainda em teste

#Nomeando para binário a coluna de sentimentos
lb = LabelBinarizer()
sentiment_data = lb.fit_transform(imdb_data['sentiment'])


#Dividindo os dados referentes a sentimentos
train_sentiments = sentiment_data[:40000]
test_sentiments = sentiment_data[40000:]

#Chamando classificador e avaliador
multinomial_nb(tf_idf_train_reviews, tf_idf_test_reviews, train_sentiments, test_sentiments)
#multinomial_nb(bow_train_reviews, bow_test_reviews, train_sentiments, test_sentiments)
