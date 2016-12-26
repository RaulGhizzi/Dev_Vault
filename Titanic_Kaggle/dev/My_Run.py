
import os
os.system('clear')


import pandas
import numpy as np 
import csv as csv
import matplotlib.pyplot as plt

from sklearn import tree


##################################Load the data to memory and return a Numpy Array#################################################
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


########################################Initialyze the data from the file#########################################################

#Data format:Matrix[rol,col]
#rol = Passager info
#col = Characteristic (i.e Sex or Name)
data, header = loader('train.csv')


########################Find all unique values for each variable and return an array with them####################################
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



############################################Load the Unique values into the memory#################################################
# access using unique_values[0,Variable][Unique_Itens]
unique_values = finduniques(header,data)

###################################################################################################################################

# Calculate the number of passagers based on the length of the columns in data[]
# number_passagers = np.size(data[0::,1].astype(np.float))
# number_passengers = np.size(data[0::,1].astype(int))
#Calculate the number of passengers based on the numer of unique IDs
number_passengers = len(unique_values[0,0])

# Split the data in survivors and deceased already converting it to an array 
alive_data = np.squeeze( np.asarray( data[data[:,1] == unique_values[0,1][1]] ) )
dead_data = np.squeeze( np.asarray( data[data[:,1] == unique_values[0,1][0]] ) )

#Calculate the number of suvivors by counting the total of "ones" in the Survived column
#number_survivors = np.sum(data[0::,1].astype(np.float))
#Get the matrix column with the data as pass it as int, convert it to an array by squeeze and get the "bitcount[1]" of "ones"
#number_survived = np.bincount(np.squeeze(np.asarray(data[:,1].astype(int))))[1]
#Get the number of survivors by getting the length of the alive_data array
number_survived = len(alive_data)

#Calculate the proportion of survivors, need to convert int to float to use fractions
proportion_survived = np.float(number_survived) / np.float(number_passengers)


#Input:
# data = Data set to be worked on
# header = The header for the given data set
# unique values = Matrix with the unique values in each columnin the data set
#col_to_keep = The columns you want to keep from the data
#col_to_change = The columns you want to rescale the values (from something to int)
#col_essential = Columns essential to the analizis, clean the rows with missing values in these columns
#Output: The new data set and the proper header for it
def data_setter(data = data, header = header, unique_values = unique_values, col_to_keep = [0, 1, 2, 4, 5, 6, 7, 9, 11], col_to_change = [4, 11], col_essential = range(12)):
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

data, header = data_setter(data, header, unique_values, [0, 1, 2, 4, 5, 6, 7, 9, 11], [4, 11], range(12))

#Imput:
#data = The data set to be splited
#axis_answer = The axis for the label (expected answer)
#axis_test = The axis to be used as parameters for the predictions
#Returns the "answer" vector and the "test" matrix
def data_spliter(data = data, axis_answer = [1], axis_test = [2,3,4,5,6,7,8]):
	answer = np.take(data, axis_answer, axis = 1).astype(np.float)
	test = np.take(data, axis_test, axis = 1)
	return answer, test

answer, test = data_spliter(data, [1], [2,3,4,5,6,7,8])

# Input:
# test = The data set to be tested
# answer = The expected answer to train the predictor with
# output: Return a Decision Tree Classifier object from sklearn
def generate_prediction_tree(test = test, answer = answer):
	predictor = tree.DecisionTreeClassifier()
	predictor = predictor.fit(test.astype(float), answer)
	return predictor

predictor = generate_prediction_tree(test, answer)


# Junk/Draft Code

# #Number of survivors by gender
# #men_survived = np.sum(men_data[0::,1].astype(np.float))
# #men_survived = np.bincount(men_data[0::,1].astype(int))[1]
# men_survived = np.bincount(np.squeeze(np.asarray(men_data[:,1].astype(int))))[1]
# #women_survived = np.sum(women_data[0::,1].astype(np.float))
# women_survived = np.bincount(women_data[0::,1].astype(int))[1]

# #female 	
# data[:,4][data[:,4] == unique_values[0,4][0]] = 0
# data[:,4][data[:,4] == unique_values[0,4][1]] = 1
# buff = np.compress((data[0::,4] == unique_values[0,4][0]).flat,data, axis=0)

# #Iterate on all the columns you pass to the function 
# for i in col:
#  	print header[i]
#  	for j in range(0,len(unique_values[0,i])):
#  		data[:,i][data[:,i] == unique_values[0,i][j]] = j

# #Split data 
# new_data = []
# new_data = np.compress((data[0::,4] == unique_values[0,4][0]).flat,data, axis=0)


# #Split the Array by gender
# #Split the data matrix based on the comparison with the words 'male' and 'female' from unique_values[0,4]
# #men_data = data[data[:,4] != 'female']
# women_data = data[data[:,4] == unique_values[0,4][0]]
# men_data = data[data[:,4] == unique_values[0,4][1]]

