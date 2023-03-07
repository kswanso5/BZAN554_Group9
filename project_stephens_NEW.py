import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


price_test = pd.read_csv('pricing_test.csv')
price_test.columns = ['sku', 'price','quantity', 'order', 'duration','category']

dummies_test = pd.get_dummies(price_test['category']).rename(columns=lambda x: 'category_' + str(x))   #create dummies
test = pd.concat([price_test, dummies_test], axis=1)  
t_cols = test.columns.values.tolist()
dummy_col_names = t_cols[6:32]
test = test.drop(['sku', 'category'], axis= 1)


response = test['quantity'].values
test_pred = test.drop(['quantity'], axis=1).values

history_loss = [] #create empty list to store loss
r2_history = [] #store r-squared scores

##Create tf model
inputs = tf.keras.layers.Input(shape=(29,), name='input')  
hidden1 = tf.keras.layers.Dense(units=4, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

model = tf.keras.Model(inputs = inputs, outputs = output)


##Read in price data 1 row at a time
for record in pd.read_csv('pricing.csv', chunksize= 1):
    if record.isna().sum().sum() > 1:
        continue   #Skip over na's
    dummy_zeros = pd.DataFrame(0, index=record.index, columns= dummy_col_names)
    dummies = pd.get_dummies(record['category']).rename(columns=lambda x: 'category_' + str(x))   #create dummies
    record = pd.concat([record, dummies], axis=1)  #merge dummies to record
    unmatched_col = dummy_zeros.columns.difference(record.columns).values.tolist() #find unmatched dummy indicators
    dummies_new = dummy_zeros[unmatched_col]
    record = pd.concat([record,dummies_new],axis= 1)
    
    if len(record.axes[1]) > 32:
        continue

    y = record['quantity'].values
    x = record.drop(['sku','category', 'quantity'], axis=1).values


    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01))

    loss = model.train_on_batch(x=x,y=y)
    history_loss.append(loss)

    model.fit(x=x ,y=y, batch_size=1, epochs=1)

    y_pred = model.predict(x)
    r2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(response)))
    r2_history.append(r2)


np.mean(response)
print(record)
    
model.evaluate(test_pred,response)

np.mean(r2_history)


y_pred = model.predict(x)
r2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(response)))
r2

r2_history.append(r2)
    

plt.plot(r2_history)
plt.show()

np.max(r2_history)

model.summary()
