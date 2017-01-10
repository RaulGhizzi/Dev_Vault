import My_prediction_lib as mplib
import numpy as np 

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split







########################################Initialyze the data from the file############################################################

#Data format:Matrix[rol,col]
#rol = Passager info
#col = Characteristic (i.e Sex or Name)
data, header = mplib.loader('train.csv')
col_answer = 1

#Retorna o novo data e o novo header, adiciona tudo no fim dos data set's
####Soma as colunas de numero de parentes
def sum_column(data, header, args = []):
	data_buff = data[:,args[0]].astype(float).A
	header_buff = header[args[0]]
	for ar in args[1:]:
		data_buff += data[:,ar].astype(float).A
		header_buff = header_buff + '+' + header[ar] 		
	# for ar in args[1:]:
	# 	header.pop(ar)
	# data = np.delete(data, args, axis = 1)
	header_buff.strip('+')
	data = np.c_[data, data_buff]
	header.append(header_buff)
	return data, header


a = data[:,2].astype(float).A
b = data[:,3].astype(float).A
c = a + b
	
data = np.c_[data,]


############################################Run a loop to check the accuracy with all the variables##################################
for i in range(col_answer + 1,len(header)):
	a, b, c = mplib.calculate_precision(data, header, [col_answer], [i], 1)
	np.corrcoef(a,b)[0,1]



# ###################################################################################################################################

# #Calculate the number of passengers based on the numer of unique IDs
# number_passengers_train = len(unique_values_train[0,0])

# # Split the data in survivors and deceased already converting it to an array 
# alive_data_train = np.squeeze( np.asarray( data_train[data_train[:,1] == unique_values_train[0,1][1]] ) )

# alive_data_test = np.squeeze( np.asarray( data_test[data_test[:,1] == unique_values_test[0,1][1]] ) )

# #Get the number of survivors by getting the length of the alive_data array
# number_survived_train = len(alive_data_train)

# #Calculate the proportion of survivors, need to convert int to float to use fractions
# proportion_survived_train = np.float(number_survived_train) / np.float(number_passengers_train)
