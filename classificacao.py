from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def multinomial_nb (model_train_reviews, model_test_reviews, train_sentiments, test_sentiments):
	mnb = MultinomialNB()
	#Aplicando o fit para tf_idf/bow dependendo do parametro passado
	mnb_model = mnb.fit(model_train_reviews, train_sentiments)
	#Prevendo o modelo para tf_idf/bow dependendo do parametro passado
	mnb_model_predict = mnb.predict(model_test_reviews)

	#Medindo a precisão (Accuracy)
	mnb_model_score = accuracy_score(test_sentiments, mnb_model_predict)
	print('Accuracy score: ')
	print(mnb_model_score)

	#Mostrando relatório das principais métricas
	mnb_model_report = classification_report(test_sentiments, mnb_model_predict, target_names=['Positive','Negative'])
	print('Classification Report: ')
	print(mnb_model_report)

	#Mostrando a matriz de confusão
	c_matrix = confusion_matrix(test_sentiments,mnb_model_predict)
	print('Confusion Matrix: ')
	print(c_matrix)
