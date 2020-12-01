from sklearn.naive_bayes import MultinomialNB


# INDETIFICA O SENTIMENTO DA CRÍTICA (CLASSIFICADOR NAIVE BAYES)
def multinomial_nb(model_train_reviews, model_test_reviews, train_sentiments, test_sentiments):
	mnb = MultinomialNB()

	# APLICANDO O FIT PARA TF_IDF/BOW DEPENDENDO DO PARÂMETRO PASSADO
	mnb_model = mnb.fit(model_train_reviews, train_sentiments)

	# PREVENDO O MODELO PARA TF_IDF/BOW DEPENDENDO DO PARÂMETRO PASSADO
	mnb_model_predict = mnb.predict(model_test_reviews)

	return mnb_model_predict