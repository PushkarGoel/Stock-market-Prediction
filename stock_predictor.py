
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("infosys2.csv")

req=df.iloc[:,4:7]
req2=df.iloc[:,8:10]

fin=pd.concat([req,req2],axis=1)


training_set=fin.iloc[:,:4].values
train_res=fin.iloc[:,4].values

from sklearn.preprocessing import MinMaxScaler
sc1=MinMaxScaler(feature_range = (0,1))
sc2=MinMaxScaler(feature_range = (0,1))
scaled_ts=sc1.fit_transform(training_set)
scaled_res=sc2.fit_transform(train_res.reshape(-1,1))

print(type(scaled_res))
print(type(scaled_ts))

scaled_ts=np.concatenate((scaled_ts,scaled_res),axis=1)

X_train=[]
y_train=[]

X_test=[]
y_test=[]

time=60;

for i in range(time,480):
    X_train.append(scaled_ts[i-time:i,0:])
    y_train.append(scaled_ts[i,4])


for i in range(480,len(scaled_ts)):
    X_test.append(scaled_ts[i-time:i,0:])
    y_test.append(scaled_ts[i,4])


X_train,y_train=np.array(X_train),np.array(y_train)
X_test,y_test=np.array(X_test),np.array(y_test)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout,Flatten
 
l=420
n_features=5

mlp = Sequential()
mlp.add(Dense(100, activation='relu',input_shape=(time,n_features)))
#model2.add(Dense(100, activation='relu'))
mlp.add(Flatten())
mlp.add(Dense(1, activation='relu'))


mlp.compile(optimizer="adam", loss="mean_squared_error")

mlp.fit(X_train, y_train, epochs=100)

resmlp=mlp.predict(X_test)

predmlp=sc2.inverse_transform(resmlp)

lstm = Sequential()
lstm.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(time, n_features)))
lstm.add(Dropout(0.2))

#model.add(LSTM(8, activation='sigmoid',return_sequences=True))
lstm.add(LSTM(8, activation='relu'))
lstm.add(Dropout(0.2))

lstm.add(Dense(1))

lstm.compile(optimizer='adam', loss='mse',metrics=['mean_squared_error'])

lstm.fit(X_train,y_train,epochs=64,batch_size=20)

reslstm=lstm.predict(X_test)

predlstm=sc2.inverse_transform(reslstm)

y_res=sc2.inverse_transform(y_test.reshape(-1,1))

losslstm=abs(y_res-predlstm)
perclstm=(losslstm/predlstm)*100

lossmlp=abs(y_res-predmlp)
percmlp=(lossmlp/predmlp)*100

avgpercmlp=sum(percmlp)/len(percmlp)
avgperclstm=sum(perclstm)/len(perclstm)

x=[i+1 for i in range(0,17)]

plt.plot(x,percmlp,color='red',label='lstm')
plt.plot(x,perclstm,color='green',label='mlp')
plt.xlabel('Day')
plt.ylabel('Percentage Difference')
plt.legend()
plt.show()


plt.plot(x,perclstm,color='red')
plt.xlabel('Day')
plt.ylabel('Percentage Difference')
plt.title('Percentage difference in actual vs predicted value')
plt.show()

#lstm graph
plt.plot(x,y_res,color='red',label='actual')
plt.plot(x,predlstm,color='green',label='Predicted')
plt.xlabel('Day')
plt.ylabel('Price of Stock')
plt.ylim((660,790))
#plt.title('Actual value vs predicted value')
plt.legend()
plt.show()


#mlp graph
plt.plot(x,y_res,color='red',label='actual')
plt.plot(x,predmlp,color='green',label='Predicted')
plt.xlabel('Day')
plt.ylabel('Price of Stock')
plt.ylim((710,760))
#plt.title('Actual value vs predicted value')
plt.legend()
plt.show()




