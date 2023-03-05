import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

history_loss = [] #create empty list to store loss
r2_history = [] #store r-squared scores

##Create tf model
inputs = tf.keras.layers.Input(shape=(4, ), name='input')  
hidden1 = tf.keras.layers.Dense(units=4, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

model = tf.keras.Model(inputs = inputs, outputs = output)



##Read in price data 1 row at a time
for record in pd.read_csv('pricing.csv', chunksize= 1):
    if record.isna().sum().sum() > 1:
        continue   #Skip over na's
    dummies = pd.get_dummies(record['category']).rename(columns=lambda x: 'category_' + str(x))   #create dummies
    record = pd.concat([record, dummies], axis=1)    #merge dummies to record
    y = record['quantity'].values
    x = record.drop(['sku','category', 'quantity'], axis=1).values
    x_array = np.array(x)   #convert x (predictors) to array

    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01))

    loss = model.train_on_batch(x=x,y=y)
    history_loss.append(loss)


#y_pred = model.predict(x)
#r2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(y)))
# #r2_history.append(r2)
    

plt.plot(history_loss)
plt.show()

plt.plot(r2_history)
plt.show()

model.summary()
