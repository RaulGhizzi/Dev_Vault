import pandas
import numpy as np 
import csv as csv


#Set the file ID for the csv, assing the Header and initialyze "data[]"
csv_file_id = csv.reader(open('train.csv', 'rb'))
header = csv_file_id.next()
data = []

#Go through the csv and stores the data in memory
for col in csv_file_id:
	data.append(col[0:])

#Convert tipe("List") to type("np.Array")
data = np.array(data)




#Calculate the number of passagers based on the length of the columns in data[]
#number_passagers = np.size(data[0::,1].astype(np.float))
number_passengers = np.size(data[0::,1].astype(int))

#Calculate the number of suvivors by counting the total of "ones" in the Survived column
#number_survivors = np.sum(data[0::,1].astype(np.float))
number_survived = np.bincount(data[0::,1].astype(int))[1]


#Calculate the proportion of survivors
proportion_survived = number_survived / number_passengers 

#Split the Array by gender
men_data = data[data[0::,4] != 'female']
women_data = data[data[::,4] == 'female']

#Number of survivors by gender
#men_survived = np.sum(men_data[0::,1].astype(np.float))
men_survived = np.bincount(men_data[0::,1].astype(int))[1]
#women_survived = np.sum(women_data[0::,1].astype(np.float))
women_survived = np.bincount(women_data[0::,1].astype(int))[1]

output = []

#Find all unique values for each variable and return an array with them

for i in range(len(header)):
	#print("Unique {}: {}").format(header[i] , np.unique(data[0::,i]))
	output.append(np.unique(data[0::,i])) 

output = np.array(output)

# Output Format:
# Header[i]: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# Output[i]: [      0      ,      1    ,     2   ,    3  ,   4  ,   5  ,    6   ,    7   ,     8   ,    9  ,    10  ,      11   ]





print("The Bug:")



