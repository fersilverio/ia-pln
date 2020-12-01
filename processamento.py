import regex as re


# CONVERTE A STRING PARA LETRAS MINÚSCULAS
def converte_minuscula(string):
	return string.lower()


# REMOVE OS ACENTOS, CEDILHAS E SIMILHARES
def remove_acentos(string):
	import unidecode as ud
	return ud.unidecode(string)


# REMOVE A PONTUAÇÃO
def remove_pontuacao(string):
	return re.sub(r'[!.,;?@#$%&:()^/|]', '', string)


# REMOVE AS TAGS HTML
def remove_html(string):
	from bs4 import BeautifulSoup
	soup = BeautifulSoup(string, 'html.parser')
	return soup.get_text()


# REMOVE OS COLCHETES E SEU CONTEÚDO INTERNO
def remove_conteudo_colchetes(string):
	return re.sub(r'\[[^]]*\]', '', string)


# CONVERTE A STRING EM LISTA TOKENIZADA
def converte_token(string):
	return string.split()


# REMOVE PALAVRAS VAZIAS (MUITO COMUNS)
def remove_palavras_vazias(lista, idioma='english'):
	try:
		from nltk.corpus import stopwords
	except:
		nltk.download('stopwords')
		from nltk.corpus import stopwords
	sw = stopwords.words(idioma)
	return [string for string in lista if string not in sw]

# REMOVE CARACTERES ESPECIAIS
def remove_caracteres_especiais(string):
	return re.sub(r'[^a-zA-z0-9\s]', '', string)


# NORMALIZA AS PALAVRAS (LEMATIZA (REDUZ AS FORMAS DE PALAVRAS A LEMAS VÁLIDOS LINGUISTICAMENTE) E STEMIZA (REDUZ PALAVRAS FLEXIONADAS OU DERIVADAS))
def normaliza(lista, metodo='all'):
	if metodo != 'stemming':
		try:
			from nltk.stem import WordNetLemmatizer
		except:
			nltk.download('wordnet')
			from nltk.stem import WordNetLemmatizer
		wn = WordNetLemmatizer()
		lista = [wn.lemmatize(string) for string in lista]
	if metodo != 'lemmatizing':
		from nltk.stem.porter import PorterStemmer
		ps = PorterStemmer()
		lista = [ps.stem(string) for string in lista]
	return lista


# CONVERTE LISTA DE TOKENS EM STRING
def converte_string(lista):
	return ' '.join(lista)


# FUNÇÃO PRINCIPAL (RETIRA IMPUREZAS QUE NÃO AJUDAM A IDENTIFICAR A CLASSE DO TEXTO)
def processa_texto(string):
	string = converte_minuscula(string)
	string = remove_html(string)
	string = remove_acentos(string)
	string = remove_pontuacao(string)
	string = remove_conteudo_colchetes(string)
	string = remove_caracteres_especiais(string)
	lista = converte_token(string)
	lista = remove_palavras_vazias(lista)
	lista = normaliza(lista)
	string = converte_string(lista)
	return string