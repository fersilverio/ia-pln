from sklearn.model_selection import train_test_split


# TERM FREQUENCY–INVERSE DOCUMENT FREQUENCY
def tf_idf(processedDf):
	from sklearn.feature_extraction.text import TfidfVectorizer
	x_train, x_test = train_test_split(processedDf, test_size=0.3)
	tfidf = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(1, 1), max_features=50000)
	tfidf_train = tfidf.fit_transform(x_train)
	tfidf_test = tfidf.transform(x_test)
	return tfidf_train, tfidf_test


# BAG OF WORDS
def bow(processedDf):
	from sklearn.feature_extraction.text import CountVectorizer
	x_train, x_test = train_test_split(processedDf, test_size=0.3)
	bow = CountVectorizer(stop_words='english', analyzer='char', ngram_range=(1, 1), max_features=50000)
	bow_train = bow.fit_transform(x_train)
	bow_test = bow.transform(x_test)
	return bow_train, bow_test

