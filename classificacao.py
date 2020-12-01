from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def multinomial_nb (tf_idf_train_reviews, tf_idf_test_reviews, train_sentiments, test_sentiments):
	mnb = MultinomialNB()
	#Aplicando o fit para tf_idf
	mnb_tfidf = mnb.fit(tf_idf_train_reviews, train_sentiments)
	#Prevendo o modelo para tf_idf
	mnb_tfidf_predict = mnb.predict(tf_idf_test_reviews)

	#Medindo a precisão (Accuracy)
	mnb_tfidf_score = accuracy_score(test_sentiments, mnb_tfidf_predict)
	print('Accuracy score: ')
	print(mnb_tfidf_score)

	#Mostrando relatório das principais métricas
	mnb_tfidf_report = classification_report(test_sentiments, mnb_tfidf_predict, target_names=['Positive','Negative'])
	print('Classification Report: ')
	print(mnb_tfidf_report)

	#Mostrando a matriz de confusão
	c_matrix = confusion_matrix(test_sentiments,mnb_tfidf_predict)
	print('Confusion Matrix: ')
	print(c_matrix)
