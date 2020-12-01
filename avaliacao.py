from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# AVALIA OS RESULTADOS GERADOS PELO CLASSIFICADOR
def avalia_resultados(test_sentiments, mnb_model_predict):
	# MEDINDO A PRECISÃO (ACCURACY)
	mnb_model_score = accuracy_score(test_sentiments, mnb_model_predict)
	print('Accuracy Score: ')
	print(mnb_model_score, '\n')

	# MOSTRANDO RELATÓRIO DAS PRINCIPAIS MÉTRICAS
	mnb_model_report = classification_report(test_sentiments, mnb_model_predict, target_names=['Positive', 'Negative'])
	print('Classification Report: ')
	print(mnb_model_report)

	# MOSTRANDO A MATRIZ DE CONFUSÃO
	c_matrix = confusion_matrix(test_sentiments, mnb_model_predict)
	print('Confusion Matrix: ')
	print(c_matrix)