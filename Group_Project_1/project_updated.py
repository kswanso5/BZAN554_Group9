import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


price_test = pd.read_csv('pricing_test.csv')
price_test.head
price_test.columns = ['sku', 'price','quantity', 'order', 'duration','category']
dummies_test = pd.get_dummies(price_test['category']).rename(columns=lambda x: 'category_' + str(x))   #create dummies
test = pd.concat([price_test, dummies_test], axis=1)  
test.head

test.head
t_array = np.array(test)
t_array.shape

response = test['quantity'].values
test_pred = test.drop(['sku','category', 'quantity'], axis=1).values

history_loss = [] #create empty list to store loss
r2_history = [] #store r-squared scores

##Create tf model
inputs = tf.keras.layers.Input(shape=(4, ), name='input')  
hidden1 = tf.keras.layers.Dense(units=4, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

model = tf.keras.Model(inputs = inputs, outputs = output)



#testrecord = pd.read_csv('pricing.csv', chunksize= 1)
#dummies_t = pd.get_dummies(testrecord['category']).rename(columns=lambda x: 'category_' + str(x))   #create dummies
#testrecord = pd.concat([testrecord, dummies_t], axis=1)    #merge dummies to record
#y_t = testrecord['quantity'].values
#x_t = testrecord.drop(['sku','category', 'quantity'], axis=1).values
#x_array_t = np.array(x_t) 

##Read in price data 1 row at a time
for record in pd.read_csv('pricing.csv', chunksize= 1):
    if record.isna().sum().sum() > 1:
        continue   #Skip over na's
    dummies = pd.get_dummies(record['category']).rename(columns=lambda x: 'category_' + str(x))   #create dummies
    record = pd.concat([record, dummies], axis=1)  #merge dummies to record
    record.shape
    y = record['quantity'].values
    x = record.drop(['sku','category', 'quantity'], axis=1).values
    x_array = np.array(x)   #convert x (predictors) to array

    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01))

    loss = model.train_on_batch(x=x,y=y)
    history_loss.append(loss)

    model.fit(x=x ,y=y, batch_size=1, epochs=1)


    
model.evaluate(test_pred,response)


#y_pred = model.predict(x)
#r2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(y)))
# #r2_history.append(r2)
    

plt.plot(history_loss)
plt.show()

plt.plot(r2_history)
plt.show()

model.summary()
