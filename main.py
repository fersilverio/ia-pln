import sys
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelBinarizer

try:
	classificador = sys.argv[1]
	if classificador not in ['bag-of-words', 'tf-idf']:
		classificador = 'tf-idf'
except:
	classificador = 'tf-idf'
print({'bag-of-words': 'Bag of Words', 'tf-idf': 'Term Frequency–Inverse Document Frequency'}[classificador], ' model selected')


'''
1. LEITURA DE DADOS
Os dados são essenciais para as análises textuais e serão obtidos através de um arquivo com 50.000 críticas de filmes divididas igualmente em duas categorias (crítica positiva e crítica negativa).
1.1. le_dados(arquivo) - Converter arquivo CSV em disco para objeto em memória.
1.2. divide_dados(dados) - C
'''
from leitura import le_dados, divide_dados

dados = le_dados('imdb-dataset.csv') # Copia o arquivo para memória
train_reviews, train_sentiments, test_reviews, test_sentiments = divide_dados(dados) # Divide os dados para treino e teste


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

dados['review'] = dados['review'].apply(processa_texto) # Aplica o pré-processamento
processed_train_reviews = dados.review[:40000] # Reviews para o treino pré-processadas
processed_test_reviews = dados.review[40000:] # Reviews para o teste pré-processadas


'''
3. EXTRAÇÃO DE CARACTERÍSTICAS
É necessário transformar o texto obtido pela etapa anterior em um vetor numérico para ser usado como entrada pelo classificador. As representações que deverão ser utilizadas são: Bag of Words e Term Frequency–Inverse Document Frequency (TF-IDF).
3.1. bow(treinamento, teste) - Simplificar o texto, desconsiderando a estrutra gramatical e até as ordenação delas, mas mantendo sua multiplicidade.
3.2. tfidf(treinamento, teste) - Indicar a importância de uma palavra em relação a um corpus linguístico.
'''
from extracao import extrai_caracteristicas

train_reviews, test_reviews = extrai_caracteristicas(processed_train_reviews, processed_test_reviews, classificador) # Transforma em matrizes (BOW ou TF-IDF) para treino e teste
lb = LabelBinarizer()
sentiment_data = lb.fit_transform(dados['sentiment']) # Nomeia a coluna de sentimentos para binário


'''
4. CLASSIFICAÇÃO
É agora que é realizada a identificação do sentimento da crítica. Alguns classificadores populares: Support Vector Machine (SVM), Árvores de Decisão, k-Nearest Neighbors (kNN), Classificador Naive Bayes.
4.1. multinomial_nb(model_train_reviews, model_test_reviews, train_sentiments, test_sentiments) - Classificar o texto assumindo que a presença de uma característica particular em uma classe não está relacionada com a presença de qualquer outro recurso.
'''
from classificacao import multinomial_nb

mnb_model_predict = multinomial_nb(train_reviews, test_reviews, train_sentiments, test_sentiments) # Identifica o sentimento


'''
5. AVALIAÇÃO DOS RESULTADOS
A métrica utilizada para avaliar os resultados gerados pelo Classificador será a F1-Score.
5.1. avalia_resultados(test_sentiments, mnb_model_predict) - Avaliar os resultados gerados pelo classificador e imprime os cálculos.
'''
from avaliacao import avalia_resultados

avalia_resultados(test_sentiments, mnb_model_predict) # Calcula a métrica F1-Score