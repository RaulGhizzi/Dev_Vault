import My_prediction_lib_Final as mplib
import numpy as np 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split







########################################Initialyze the data from the file############################################################

#Data format:Matrix[rol,col]
#rol = Passager info
#col = Characteristic (i.e Sex or Name)
data_train, header_train = mplib.loader('train.csv')
col_answer = [1]
col_variables_train = [4,5]

data_test, header_test = mplib.loader('test.csv')
col_variables_test = [3,4]

predictor = mplib.train_predictor(data_train, header_train, col_answer, col_variables_train)
prediction = mplib.make_prediction(predictor, data_test, header_test, col_variables_test)




############################################Run a loop to check the accuracy with all the variables##################################
for i in range(col_answer + 1,len(header)):
	a, b, c = mplib.calculate_precision(data, header, [col_answer], [i], 1)
	np.corrcoef(a,b)[0,1]



