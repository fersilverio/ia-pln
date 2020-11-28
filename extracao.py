from sklearn.model_selection import train_test_split


# TERM FREQUENCY–INVERSE DOCUMENT FREQUENCY
def tf_idf(string):
	from sklearn.feature_extraction.text import TfidfVectorizer
	x_train, x_test = train_test_split(string, test_size=0.3)
	vectorizer = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(1, 1), max_features=50000)
	vectorizer.fit_transform(x_train)
	vectorizer.transform(x_test)
	return vectorizer.transform(x_train).todense()


# BAG OF WORDS
def bow(processedDf):
	from sklearn.feature_extraction.text import CountVectorizer
	x_train, x_test = train_test_split(processedDf, test_size=0.3)
	bow = CountVectorizer(stop_words='english', analyzer='char', ngram_range=(1, 1), max_features=50000)
	bow_train = bow.fit_transform(x_train)
	bow_test = bow.transform(x_test)
	#return vectorizer.transform(x_train).todense()
	return bow_train, bow_test







''' Fazer desse jeito vai contra a ideia do role - fernando
############################################### REVISAR
def extracao(string):
	count_train = tf_idf(string)
	bag_of_words = bow(string)
	pass
'''