import My_Run_v3.0


########################################Initialyze the data from the file#########################################################

#Data format:Matrix[rol,col]
#rol = Passager info
#col = Characteristic (i.e Sex or Name)
data_train, header_train = loader('train.csv')


############################################Load the Unique values into the memory#################################################
# access using unique_values[0,Variable][Unique_Itens]
unique_values_train = finduniques(header_train,data_train)


data_train, header_train = data_setter(data_train, header_train, unique_values_train, [0, 1, 2, 4, 5, 6, 7, 9, 11], [4, 11], range(data_train.shape[1]))


answer_train, variables_train = data_spliter(data_train, [1], [2,3,4,5,6,7,8])

x_train, x_test, y_train, y_test = train_test_split(variables_train, answer_train,test_size = 0.5)

predictor = generate_prediction_tree(x_train, y_train)

predictor = predictor.predict(x_test)

print accuracy_score(y_test, predictor)

###################################################################################################################################

#Calculate the number of passengers based on the numer of unique IDs
number_passengers_train = len(unique_values_train[0,0])

# Split the data in survivors and deceased already converting it to an array 
alive_data_train = np.squeeze( np.asarray( data_train[data_train[:,1] == unique_values_train[0,1][1]] ) )

alive_data_test = np.squeeze( np.asarray( data_test[data_test[:,1] == unique_values_test[0,1][1]] ) )

#Get the number of survivors by getting the length of the alive_data array
number_survived_train = len(alive_data_train)

#Calculate the proportion of survivors, need to convert int to float to use fractions
proportion_survived_train = np.float(number_survived_train) / np.float(number_passengers_train)
