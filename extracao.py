from sklearn.model_selection import train_test_split


# TERM FREQUENCY–INVERSE DOCUMENT FREQUENCY
def tf_idf(string):
	from sklearn.feature_extraction.text import TfidfVectorizer
	x_train, x_test = train_test_split(string, test_size=0.2, random_state=42)
	vectorizer = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(1, 1), max_features=50000)
	vectorizer.fit(x_train)
	return vectorizer.transform(x_train).todense()


# BAG OF WORDS
def bow(string):
	from sklearn.feature_extraction.text import CountVectorizer
	x_train, x_test = train_test_split(string, test_size=0.2, random_state=42)
	vectorizer = CountVectorizer(stop_words='english', analyzer='char', ngram_range=(1, 1), max_features=50000)
	# vectorizer.fit(x_train) # COUNT TRAIN
	vectorizer.transform(x_train) # BAG OF WORDS
	return vectorizer.transform(x_train).todense()


############################################### REVISAR
def extracao(string):
	count_train = tf_idf(string)
	bag_of_words = bow(string)
	pass