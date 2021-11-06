import math
import pandas_datareader as web
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime,timedelta



def predict(stock: str):
    plt.style.use ('fivethirtyeight')


    #fecha de hoy
    today = str(datetime.today().year)+"-"+str(datetime.today().month)+"-"+str(datetime.today().day)


    df = web.DataReader(stock,data_source='yahoo', start='2012-01-01', end = today)

    plt.figure(figsize=(16,8))
    plt.title('Close price history')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)



    dataset = df.filter(['Close']).values

    train_data_len = math.ceil(len(dataset) * 0.8)


    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(dataset)


    train_data = scaled_data[0:train_data_len,:]

    x_train = []
    y_train = []

    for i in range (100,len(train_data)):
        x_train.append(train_data[i-100:i, 0])
        y_train.append(train_data[i , 0])




    #convertir listas a arreglos numpy
    x_train,y_train = np.array(x_train), np.array(y_train)


    #Reshape
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


    model = Sequential()

    model.add(LSTM(50,return_sequences=True, input_shape = (x_train.shape[1],1 )))
    model.add(LSTM(50,return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mean_squared_error')

    model.fit(x_train,y_train,batch_size=1,epochs=1)


    test_data = scaled_data[train_data_len - 100 : ,:]

    x_test = []
    y_test = dataset[train_data_len:, :]
    for i in range (100,len(test_data)):
        x_test.append(test_data[i-100:i, 0])



    x_test = np.array(x_test)


    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)


    rmse = np.sqrt(np.mean(predictions - y_test)**2)


    train = df.filter(['Close'])[:train_data_len]
    valid = df.filter(['Close'])[train_data_len:]
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


    apple_quote=web.DataReader(stock,data_source='yahoo',start='2012-01-01',end=today)
    new_df = apple_quote.filter(['Close'])

    last_100_days = new_df[-100:].values
    last_100_days_scaled = scaler.fit_transform(last_100_days)


    X_test = []

    X_test.append(last_100_days_scaled)
    X_test = np.array(X_test)


    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

    pred_price = model.predict(X_test)

    pred_price =scaler.inverse_transform(pred_price)
    print(pred_price)