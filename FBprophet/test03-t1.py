#import packages 
import pandas as pd
import numpy as np
from fbprophet import Prophet
#to plot within notebook
import matplotlib.pyplot as plt
#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
#read the file
df = pd.read_csv('0050.TW.csv')
data=df
new_data = pd.DataFrame(index=range(0,len(df)),columns=['ds', 'y'])

for i in range(0,len(data)):
    new_data['ds'][i] = data['ds'][i]
    new_data['y'][i] = data['y'][i]
new_data.rename(columns={'y': 'y', 'ds': 'ds'}, inplace=True)
new_data['ds'] = pd.to_datetime(new_data['ds'], format='%Y-%m-%d')
#new_data.index = new_data['ds']
trainA =int(len (new_data) * 0.67)
validA  =len(new_data) - trainA
train = new_data[:trainA]
valid =  new_data[trainA:]
#fit the model
model = Prophet()
model.fit(new_data)
#predictions
Train_close_prices = model.make_future_dataframe(periods=validA,freq = 'D',  include_history = False)
forecast = model.predict(Train_close_prices)
#print(forecast['yhat'],valid)
test_valid = valid
test_valid.sort_values(by=['ds'])
test_valid.reset_index(inplace=True)
#rmse
forecast_valid = forecast['yhat'].round(decimals=1)
rms=np.sqrt(np.mean(np.power((np.array(test_valid['y'])-np.array(forecast_valid)),2)))
rms.round(decimals=2)
print("RMSE: [ ", rms , " ]")
#plt
plt.plot(test_valid['y'])
plt.plot(forecast_valid)
plt.show()
#test_valid['y'].to_csv('test_valid.csv')
#forecast_valid.to_csv('forecast_valid.csv')
