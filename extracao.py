from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# TERM FREQUENCY–INVERSE DOCUMENT FREQUENCY
# x_train e x_test são relativos a review
def tf_idf(x_train,x_test):	
	tfidf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1, 3))
	tfidf_train = tfidf_vectorizer.fit_transform(x_train)
	tfidf_test = tfidf_vectorizer.transform(x_test)
	return tfidf_train, tfidf_test

# BAG OF WORDS
# x_train e x_test são relativos a review
def bow(x_train,x_test):
	bow = CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
	bow_train = bow.fit_transform(x_train)
	bow_test = bow.transform(x_test)
	return bow_train, bow_test

