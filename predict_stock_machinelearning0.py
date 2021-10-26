import math
import pandas_datareader as web
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


plt.style.use ('fivethirtyeight')

df = web.DataReader('ADA-USD',data_source='yahoo', start='2012-01-01', end = '2021-12-31')
df

df.shape



plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
#plt.ylabel('Close Price USD ($)',18)


# Create a new dataframe with only the Close column

btc_close=df.filter(['Close'])
#Convert the dataframe to a numpy array

dataset = btc_close.values

# get the number of rows to train the model on

training_data_len = math.ceil(len(dataset) * 0.8)

training_data_len




#Scale the data

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


#create the training data set
#Create the scaled data set

train_data = scaled_data[0:training_data_len,:]
#Split the data into x_train and y_train data sets

x_train = []
y_train = []

for i in range (100,len(train_data)):
    x_train.append(train_data[i-100:i, 0])
    y_train.append(train_data[i , 0])



#conver the c_train and y_train to numpy arrays


x_train,y_train = np.array(x_train), np.array(y_train)
#Reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape




#Build the LSTM model
model = Sequential()

model.add(LSTM(50,return_sequences=True, input_shape = (x_train.shape[1],1 )))
model.add(LSTM(50,return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))



#compile the model
model.compile(optimizer='adam',loss='mean_squared_error')






#train the model
model.fit(x_train,y_train,batch_size=1,epochs=1)



#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2003

test_data = scaled_data[training_data_len - 100 : ,:]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range (100,len(test_data)):
    x_test.append(test_data[i-100:i, 0])

#convert the data to numpy array

x_test = np.array(x_test)
type(x_test)
x_test.shape



#Reshape the Data
#print(x_test)
for i in range (0,60):
    temp = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    #Get the models predicted price values
    predictions = model.predict(temp)

    np.append(x_test,np.append(x_test[-1][1:].copy(),predictions))
    #print(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


#Get root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)



#Plot the data
train = btc_close[:training_data_len]
valid = btc_close[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel=('Close Price USD ($)',18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'], loc = 'lower right')
plt.show()




valid

#Get the quote
apple_quote=web.DataReader('ADA-USD',data_source='yahoo',start='2012-01-01',end='2021-09-23')
#create a new Dataframe
new_df = apple_quote.filter(['Close'])
#Get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-100:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#create and empty list

X_test = []
#append the last 60days
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
type(X_test)







X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))


#Get the predicted scaled price




pred_price = model.predict (X_test)

#undo the scaling
pred_price =scaler.inverse_transform(pred_price)
#print(pred_price)




#apple_quote2=web.DataReader('DOGE-USD',data_source='yahoo',start='2021-09-24',end='2021-09-24')
#print(apple_quote2['Close'][0])


apple_quote2=web.DataReader('ADA-USD',data_source='yahoo',start='2021-09-24',end='2021-09-24')
print("Real: "+str(apple_quote2['Close'][0]))

print("prec")



porc = (dif*100)/apple_quote2['Close'][0]
print(str(porc)+"%")
