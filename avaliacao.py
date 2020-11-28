from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

################ REVISAR
def avaliacao(parametros):
	matriz_confusao = metrics.confusion_matrix(y_t, y_p, labels=[1, 0])

	print('Matrix de Confusão')
	print(matriz_confusao)

	pd.set_option('display.float_format', lambda x: '%.3f' % x)

	df_cm = pd.DataFrame(matriz_confusao, range(2), range(2))
	sn.set(font_scale=1.4) # for label size
	labels = ['Spam', 'Ham']
	sn.heatmap(df_cm, annot=True, annot_kws={"size": 16, }, fmt='g', xticklabels=labels, yticklabels=labels) # font size
	plt.show()

	print(f'Acuracia: {metrics.accuracy_score(y_t, y_p)}')
	print(f'Precisao: {metrics.precision_score(y_t, y_p)}')
	print(f'Revocação: {metrics.recall_score(y_t, y_p)}')
	print(f'F1-Score: {metrics.f1_score(y_t, y_p)}')