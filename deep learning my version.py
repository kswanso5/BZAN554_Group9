import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


price_test = pd.read_csv(r'C:\Users\Chase\Downloads\pricing_test.csv')


possibilites = [1,2 ,3 ,4 ,5 ,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 ]
#atlist = pd.read_csv(r'C:\Users\Chase\Downloads\categorylist.csv').sort_values(by='category_values')
#sorted_catlist = catlist['category_values'].values.tolist()
price_test.columns = ['sku', 'price','quantity', 'order', 'duration','category']

for row in price_test:
#dummies_test = pd.get_dummies(price_test['category']).rename(columns=lambda x: 'category_' + str(x))   #create dummies
    exists = row[0].tolist()
    difference = pd.Series([item for item in possibilites if item not in exists])

    target = price_test["category"].append(pd.Series(difference))
    target = target.reset_index(drop=True)
    dummies = pd.get_dummies(
    target)
    dummies = dummies.drop(dummies.index[list(range(len(dummies)-len(difference), len(dummies)))])
    dummiesdf = pd.DataFrame(dummies)
    price_test = price_test.merge(dummiesdf, how = 'cross') 

row

#test = pd.concat([price_test, dummies_test], axis=1)  



t_array = np.array(test)
t_array.shape

response = test['quantity'].values
test_pred = test.drop(['sku','category', 'quantity'], axis=1).values

history_loss = [] #create empty list to store loss
r2_history = [] #store r-squared scores

##Create tf model
inputs = tf.keras.layers.Input(shape=(35,1 ), name='input')  
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
for record in pd.read_csv(r'C:\Users\Chase\Downloads\pricing.csv', chunksize= 1):
    if record.isna().sum().sum() > 1:
        continue   #Skip over na's
    exists = record["category"].tolist()
   # dummies = pd.DataFrame([catlist["category_values"].eq(record['category'].values[0]).values.tolist()], columns=sorted_catlist)
    difference = pd.Series([item for item in possibilites if item not in exists])
    target = record["category"].append(pd.Series(difference))
    target = target.reset_index(drop=True)
    dummies = pd.get_dummies(
    target)
    dummies = dummies.drop(dummies.index[list(range(len(dummies)-len(difference), len(dummies)))])
    dummiesdf = pd.DataFrame(dummies)
   # dummies = pd.get_dummies(record, columns = ['category'], prefix='category') #create dummies
    record = record.merge(dummiesdf, how = 'cross')  #merge dummies to record
    y = record['quantity'].values
    x = record.drop(['sku','category', 'quantity'], axis=1).values
    x_array = np.array(x)   #convert x (predictors) to array

    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01))

    loss = model.train_on_batch(x=x,y=y)
    history_loss.append(loss)

    model.fit(x=x ,y=y, batch_size=1, epochs=1)

dummies.index   
record.index 
model.evaluate(test_pred,response)


#y_pred = model.predict(x)
#r2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(y)))
# #r2_history.append(r2)
    

plt.plot(history_loss)
plt.show()

plt.plot(r2_history)
plt.show()

model.summary()