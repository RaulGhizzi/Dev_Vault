
import os
os.system('clear')

import pandas
import numpy as np 
import csv as csv

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split




##################################Load the data to memory and return a Numpy Matrix#################################################
def loader(file_name = 'train.csv'):
	#Set the file ID for the csv, assing the Header and initialyze "data[]"
	csv_file_id = csv.reader(open(file_name, 'rb'))
	header = csv_file_id.next()
	data = []
	#Go through the csv and stores the data in memory
	for col in csv_file_id:
		data.append(col[0:])
	#Convert tipe("List") to type("np.Array")
	data = np.matrix(data)
	return data, header


########################Find all unique values for each variable and return an array with them#####################################
# Output Format:
# Header[i]: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# Output[i]: [      0      ,      1    ,     2   ,    3  ,   4  ,   5  ,    6   ,    7   ,     8   ,    9  ,    10  ,      11   ]
def finduniques(header, data):
	output = []
	for i in range(len(header)):
		#print("Unique {}: {}").format(header[i] , np.unique(data[0::,i]))
		output.append(np.unique(data[:,i].A.astype(type(data[0,i]))))
	output = np.matrix(output)
	return output

##################Filter and reshape the data set matrix, used to treat the data before using it################################### 
#Input:
# data = Data set to be worked on
# header = The header for the given data set
# unique values = Matrix with the unique values in each columnin the data set
#col_to_keep = The columns you want to keep from the data
#col_to_change = The columns you want to rescale the values (from something to int)
#col_essential = Columns essential to the analizis, clean the rows with missing values in these columns
#Output: The new data set and the proper header for it
def data_setter(data , header , unique_values , col_to_keep, col_to_change, col_essential):
 	#Get the lenght of the unique_values 
 	size = data.shape[1]
 	resize = range(size)
 	to_del = []
 	header_buff = []
 	for l in col_to_keep:
 		header_buff.append(header[l])
 	#Split the data keeping only the columns you passed in col 
 	for k in resize:
 		if k not in col_to_keep[:]:
	 		resize[k] = False
	 	else:
	 		resize[k] = True	
 	#Goes through the columns with non int values and reset the values
 	#Male = 1, Female = 0 || Embarked '' = 0, 'C' = 1, 'Q' = 2, 'S' = 3
 	for i in col_to_change: 
 		for j in range(len(unique_values[0,i])):
 			data[:,i][data[:,i] == unique_values[0,i][j]] = j 	
 	#Marks the empity or zero elements and deletes the lines
 	for s in col_essential:
 		for t in range(data.shape[0]):
 			if data[t,s] == '' or data[t,s] == 0:
 				to_del.append(t) 		
 	data = np.delete(data, to_del, axis = 0) 	
 	#Use the bollean resize to cut the columns you don`t want
 	data = np.compress(resize, data, axis=1)	 	
	return data, header_buff	

###############################################Split the data to be used in the prediction#########################################
#Imput:
#data = The data set to be splited
#axis_answer = The axis for the label (expected answer)
#axis_variables = The axis to be used as parameters for the predictions, the problem variables.
#Returns the "answer" vector and the "variables" matrix
def data_spliter(data, axis_answer, axis_variables):
	answer = np.take(data, axis_answer, axis = 1).astype(np.float)
	variables = np.take(data, axis_variables, axis = 1)
	return answer, variables


##########################################Generate a Prediction Tree Object with the given data###################################
# Input:
# test = The data set to be tested
# answer = The expected answer to train the predictor with
# output: Return a Decision Tree Classifier object from sklearn
def generate_prediction_tree(variables, answer):
	predictor = tree.DecisionTreeClassifier()
	predictor = predictor.fit(variables.astype(float), answer)
	return predictor



###########################################Create the predictor to be used for the test data ####################################
def make_predictor():
	data, header = loader(file_name)
	unique_values = finduniques(header, data)
	data, header = data_setter(data, header, unique_values, col_to_keep, col_to_change, col_essential)
	answer, variables = data_spliter(data, axis_answer, axis_variables)
	predictor = generate_prediction_tree(variables, answer)
	return predictor


########################################Initialyze the data from the file#########################################################

#Data format:Matrix[rol,col]
#rol = Passager info
#col = Characteristic (i.e Sex or Name)
data_train, header_train = mplib.loader('train.csv')


############################################Load the Unique values into the memory#################################################
# access using unique_values[0,Variable][Unique_Itens]
unique_values_train = mplib.finduniques(header_train,data_train)


data_train, header_train = mplib.data_setter(data_train, header_train, unique_values_train, [0, 1, 2, 4, 5, 6, 7, 9, 11], [4, 11], range(data_train.shape[1]))


answer_train, variables_train = mplib.data_spliter(data_train, [1], [2,3,4,5,6,7,8])

x_train, x_test, y_train, y_test = train_test_split(variables_train, answer_train,test_size = 0.5)

predictor = mplib.generate_prediction_tree(x_train, y_train)

predictor = predictor.predict(x_test)

print accuracy_score(y_test, predictor)
