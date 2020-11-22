import regex as re
import numpy as np
from bs4 import BeautifulSoup
import unidecode
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

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
def _removeStopWords (string):
    print('remove palavras vazias')

# Normalização
def _normalize (string):
    print('normalizando')



def text_processing (string):
    string = _removeHtml(string)
    string = _toLower(string)
    string = _removeAcentos(string)
    string = _removePontuacao
    string = _removeBracketsContent(string)
    return string




string = input()
'''
print(string)
#a = text_processing(string)
a = _removePontuacao(string)
print(a)
'''

a = _tokenize(string)
print(a)