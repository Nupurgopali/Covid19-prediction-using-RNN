# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('covid_india_train.csv')
training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 20 timesteps and t+1 output
X_train = []
y_train = []
for i in range(20, 125):
    X_train.append(training_set_scaled[i-20:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 3,activation='sigmoid',input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 220, batch_size = 32)

# Joining the train and test dataset under the real no. of covid cases in India
dataset_test = pd.read_csv('covid_india_test.csv')
test_set = dataset_test.iloc[:,1:2].values
real_covid_cases = np.concatenate((training_set[0:125], test_set), axis = 0)

# Getting the predicted number of cases
scaled_real_covid_cases= sc.fit_transform(real_covid_cases)
inputs = []
for i in range(126, 136):
    inputs.append(scaled_real_covid_cases[i-20:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

predicted_covid_cases = regressor.predict(inputs)

predicted_covid_cases = sc.inverse_transform(predicted_covid_cases)
predicted_covid_cases=predicted_covid_cases.astype('int64')

# Visualising the results
plt.plot(test_set, color = 'red', label = 'Real Covid Cases')
plt.plot(predicted_covid_cases, color = 'blue', label = 'Predicted Covid Cases')
plt.title('Covid Prediction')
plt.xlabel('Time')
plt.ylabel('Predition of covid cases')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(test_set,predicted_covid_cases))

#predict the next 2 days case
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
real_covid_cases_scaled = sc.fit_transform(real_covid_cases)
input_next_pred=real_covid_cases_scaled.reshape(-1,1)
list_input=list(input_next_pred)
from numpy import array

lst_output=[]
n_steps=135
i=0
while(i<2):
    
    if(len(list_input)>135):
        #print(temp_input)
        input_next_pred=np.array(list_input[1:])
        #print("{} day input {}".format(i,x_input))
        input_next_pred=input_next_pred.reshape(1,-1)
        input_next_pred = input_next_pred.reshape((1, n_steps, 1))
        #print(x_input)
        y_pred = regressor.predict(input_next_pred, verbose=0)
        y_pred = sc.inverse_transform(y_pred)
        y_pred=y_pred.astype('int64')
        
        #print("{} day output {}".format(i,yhat))
        list_input.extend(y_pred[0].tolist())
        list_input=list_input[1:]
        #print(temp_input)
        lst_output.extend(y_pred.tolist())
        i=i+1
    else:
        input_next_pred= input_next_pred.reshape((1, n_steps,1))
        y_pred = regressor.predict(input_next_pred, verbose=0)
        y_pred = sc.inverse_transform(y_pred)
        y_pred=y_pred.astype('int64')
        print(y_pred[0])
        list_input.extend(y_pred[0].tolist())
        print(len(list_input))
        lst_output.extend(y_pred.tolist())
        i=i+1
    

print("Total covid cases in India on 13 June will be apporx: ",lst_output[0])
print("Total covid cases in India on 14 June will be apporx: ",lst_output[1])

day_new=np.arange(1,136)
day_pred=np.arange(136,138)
day_new=day_new.reshape(-1,1)

plt.plot(day_new,real_covid_cases,color = 'blue', label = 'Real Covid Cases')
plt.plot(day_pred,lst_output,color = 'green', label = 'Next 2 days cases prediction')
plt.title('Next 2 days covid prediction')
plt.xlabel('Time')
plt.ylabel('Predition of new covid cases')
plt.legend()
plt.show()





