import pandas
import numpy as np 
import csv as csv

import os
os.system('clear')

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



#Split the Array by gender
#Split the data matrix based on the comparison with the words 'male' and 'female' from unique_values[0,4]
#men_data = data[data[:,4] != 'female']
women_data = data[data[:,4] == unique_values[0,4][0]]
men_data = data[data[:,4] == unique_values[0,4][1]]

#var_to_count is the variable you want to count (i.e )
def spliter(data = data, unique_values = unique_values, var_to_count = 1, var_to_check):
	buff = []
	for i in len(unique_values[0,var_to_check]):



#Number of survivors by gender
#men_survived = np.sum(men_data[0::,1].astype(np.float))
#men_survived = np.bincount(men_data[0::,1].astype(int))[1]
men_survived = np.bincount(np.squeeze(np.asarray(men_data[:,1].astype(int))))[1]
#women_survived = np.sum(women_data[0::,1].astype(np.float))
women_survived = np.bincount(women_data[0::,1].astype(int))[1]










print("The Bug:")



