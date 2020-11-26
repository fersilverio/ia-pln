import regex as re
import numpy as np
from bs4 import BeautifulSoup
import unidecode
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Deixa toda a string para letras minusculas
def _toLower (string):
    string = string.lower()
    return string

# Remove as acentos , cedilhas e similares
def _removeAcentos (string):
    string = unidecode.unidecode(string)
    return string

#Remove a pontuacao
def _removePontuacao (string):
    string = re.sub('[!.,;?@#$%&:()^/|]', '', string)
    return string

# Remove as tags html
def _removeHtml (string):
    soup = BeautifulSoup(string, "html.parser")
    string = soup.get_text() 
    return string

# Remove os brackets e seu conteudo interno
def _removeBracketsContent(string):
    string = re.sub('\[[^]]*\]', '', string) 
    return string


# Tokenização
def _tokenize (string):
    tokenized_string = string.split()
    return tokenized_string


# Remoção de palavras-vazias (muito comuns)
def _removeStopWords (tokenized_string, language='english'):
    stopword = stopwords.words(language)
    text = [word for word in tokenized_string if word not in stopword]
    return text
'''
# Normalização
def _normalize (tokenized_string):
    wn = WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokenized_string]
    return text
'''
def _normalize(tokenized_string):
	ps = PorterStemmer()
	text = [ps.stem(word) for word in tokenized_string]
	return text



def _backToText(string):
    detokenized_string = TreebankWordDetokenizer().detokenize(string)
    return detokenized_string


# Realização pro processamento textual requerido
def text_processing (string):
    string = _removeHtml(string)
    string = _toLower(string)
    string = _removeAcentos(string)
    string = _removePontuacao(string)
    string = _removeBracketsContent(string)
    string = _tokenize(string)
    string = _removeStopWords(string)
    string = _normalize(string)
    string = _backToText(string)
    return string




#string = input()
'''
print(string)
#a = text_processing(string)
a = _removePontuacao(string)
print(a)
'''

#a = text_processing("Son of Tamriel you are the fucking Dragonborn")
#print(a)