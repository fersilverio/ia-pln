# BAG OF WORDS
def bow(x_train, x_test):
	from sklearn.feature_extraction.text import CountVectorizer
	bow = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
	# X_TRAIN E X_TEST SÃO RELATIVOS A REVIEW
	bow_train = bow.fit_transform(x_train)
	bow_test = bow.transform(x_test)

	return bow_train, bow_test


# TERM FREQUENCY–INVERSE DOCUMENT FREQUENCY
def tf_idf(x_train, x_test):
	from sklearn.feature_extraction.text import TfidfVectorizer
	tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 3))
	# X_TRAIN E X_TEST SÃO RELATIVOS A REVIEW
	tfidf_train = tfidf_vectorizer.fit_transform(x_train)
	tfidf_test = tfidf_vectorizer.transform(x_test)

	return tfidf_train, tfidf_test


# FUNÇÃO PRINCIPAL
def extrai_caracteristicas(x_train, x_test, metodo='bag-of-words'):
	if metodo == 'bag-of-words':
		return bow(x_train, x_test)
	else:
		return tf_idf(x_train, x_test)