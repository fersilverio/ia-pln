from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

def classificacao(corpus, y):
	split = 5
	kf = KFold(n_splits=split, shuffle=True, random_state=0)
	y_t = []
	y_p = []

	for train_index, test_index in tqdm(kf.split(corpus), total=split):
		# print("TRAIN:", train_index, "TEST:", test_index)
		
		x_train, x_test = corpus[train_index], corpus[test_index]
		vectorizer = TfidfVectorizer(ngram_range=(1, 1))
		# vectorizer = CountVectorizer()
		
		x_train = vectorizer.fit_transform(x_train)
		x_test = vectorizer.transform(x_test)
		
		# print(x_train.shape, x_test.shape)

		y_train, y_test = y[train_index], y[test_index]

		clf = SVC(kernel='linear') # clf = LinearSVC()
		clf.fit(x_train, y_train.ravel())
		y_pred = clf.predict(x_test)

		y_t.extend(y_test)
		y_p.extend(y_pred)

	return y_t, y_p