import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sklearn.ensemble
import tracemalloc
import time
import eli5
from sklearn.inspection import partial_dependence
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import math
from sklearn.ensemble import RandomForestRegressor




price_test = pd.read_csv(r'C:\Users\Chase\Downloads\pricing_test.csv') #import test file
mms = StandardScaler() #create scaler

possibilites = [0,1,2 ,3 ,4 ,5 ,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 ] #create list of possible categories, to be used later in get dummies
price_test.columns = ['sku', 'price','quantity', 'order', 'duration','category'] #manually adding column names



##CREATING FAKE ROWS TO ADD CATEGORIES UNSEEN IN TESTING DATA BE IN TRAINING DATA

#NOT PROBABLY THE BEST WAY TO DO IT, BUT OTHER METHODS WEREN'T WORKING

df10 = pd.DataFrame({'sku' : 136061, 'price': 0.01, 'quantity': 1, 'order':1361134857, 'duration':46, 'category':0}, columns = ['sku', 'price','quantity', 'order', 'duration','category'], index = [0])
df3 = pd.DataFrame({'sku' : 136061, 'price': 0.01, 'quantity': 1, 'order':1361134857, 'duration':46, 'category':4},  columns = ['sku', 'price','quantity', 'order', 'duration','category'], index = [0])
df4 = pd.DataFrame({'sku' : 136061, 'price': 0.01, 'quantity': 1, 'order':1361134857, 'duration':46, 'category':13}, columns = ['sku', 'price','quantity', 'order', 'duration','category'], index = [0])
df5 = pd.DataFrame({'sku' : 136061, 'price': 0.01, 'quantity': 1, 'order':1361134857, 'duration':46, 'category':16}, columns = ['sku', 'price','quantity', 'order', 'duration','category'], index = [0])
df6 = pd.DataFrame({'sku' : 136061, 'price': 0.01, 'quantity': 1, 'order':1361134857, 'duration':46, 'category':20}, columns = ['sku', 'price','quantity', 'order', 'duration','category'], index = [0])
df7 = pd.DataFrame({'sku' : 136061, 'price': 0.01, 'quantity': 1, 'order':1361134857, 'duration':46, 'category':23}, columns = ['sku', 'price','quantity', 'order', 'duration','category'], index = [0])
df8 = pd.DataFrame({'sku' : 136061, 'price': 0.01, 'quantity': 1, 'order':1361134857, 'duration':46, 'category':25}, columns = ['sku', 'price','quantity', 'order', 'duration','category'], index = [0])
price_test = price_test.append(df10, ignore_index = True)
price_test = price_test.append(df3, ignore_index = True)
price_test = price_test.append(df4, ignore_index = True)
price_test = price_test.append(df5, ignore_index = True)
price_test = price_test.append(df6, ignore_index = True)
price_test = price_test.append(df7, ignore_index = True)
price_test = price_test.append(df8, ignore_index = True)


test = pd.get_dummies(price_test, columns = ['category']) #create dummies

test[['price','quantity', 'order', 'duration']] = mms.fit_transform(test[['price','quantity', 'order', 'duration']])
response = test['quantity'].values # Y VALUES
test_pred = test.drop(columns =['sku', 'quantity']).values # X VALUES




history_loss = [] #create empty list to store loss
r2_history = [] #store r-squared scores
mse_mean_history = [] #store MSE MOVING AVERAGE


##Create tf model
inputs = tf.keras.layers.Input(shape=(36,1 ), name='input')  
hidden1 = tf.keras.layers.Dense(units=4, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

model = tf.keras.Model(inputs = inputs, outputs = output)

# MSE MOVING AVERAGE TRACKED WITH A 3 ROW WINDOW
window_size = 3
mse_mean = tf.keras.metrics.Mean()


tracemalloc.start() #STARTS MEMORY ALLOCATION
start = time.time() #STARTS TIME
##Read in price data 1 row at a time
for record in pd.read_csv(r'C:\Users\Chase\Downloads\pricing.csv', chunksize= 1):
    if record.isna().sum().sum() > 1:
        continue   #Skip over na's
    exists = record["category"].tolist()
    difference = pd.Series([item for item in possibilites if item not in exists])
    target = record["category"].append(pd.Series(difference))
    target = target.reset_index(drop=True)
    dummies = pd.get_dummies(
    target)
    dummies = dummies.drop(dummies.index[list(range(len(dummies)-len(difference), len(dummies)))])
    dummiesdf = pd.DataFrame(dummies)
    record = record.merge(dummiesdf, how = 'cross')  #merge dummies to record
    record[['price']] = record[['price']] - 0.7388322
    record[['price']] = record[['price']] / 1.952867
    record[['quantity']] = record[['quantity']] - 23.02006
    record[['quantity']] = record[['quantity']] / 47.19862
    record[['order']] = record[['order']] - 1308728285
    record[['order']] = record[['order']] / 52768115
    record[['duration']] = record[['duration']] - 147.7455
    record[['duration']] = record[['duration']] / 165.814
    y = record['quantity'].values
    x = record.drop(['sku','category', 'quantity'], axis=1).values
    y_pred = model.predict(x)
    r2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(response)))
    r2_history.append(r2)
    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01))
    loss = model.train_on_batch(x=x,y=y)
    history_loss.append(loss)
    model.fit(x=x ,y=y, batch_size=1, epochs=1)
    mse_mean.update_state(loss)
    if mse_mean.count >= window_size:
            moving_avg_mse = mse_mean.result()
            tracker = moving_avg_mse.numpy()
            mse_mean_history.append(tracker)
            mse_mean.reset_states()






### VARIABLE IMPORTANCE PLOTS
#yhat = model.predict(test_pred)
#performance_before = np.corrcoef(response,yhat)[0,1]
#performance_before

#step 3

#importance = list()
#for ind in [1,2]:
    #3.1
    #cte_X_cp = np.copy(test_pred)
    #variable = np.random.permutation(np.copy(test_pred[:,ind]))
    #note: np.random.pemutation already makes a copy, but including
    #np.copy for explicitness
    #cte_X_cp[:,ind] = variable
    #3.2
    #yhat = model.predict(cte_X_cp)
    #performance_after = np.corrcoef(repsonse,yhat)[0,1]
    #3.3
    #importance.append(performance_before - performance_after)

#plt.figure()
#plt.title("Variable Importance")
#plt.bar(range(test_pred.shape[1]), importances[indices],
#color="r", yerr=std[indices], align="center")
#plt.xticks(range(test_pred.shape[1]), test_pred_test.columns[indices], rotation='vertical')
#plt.xlim([-1, test_pred.shape[1]])
#plt.show()


### CREATION OF PARTIAL DEPENDANCE PLOTS
#ind = 3 #manually changed index
#v = np.unique(test_pred[1:100,ind])
#test_pred[1:25,ind]
#means = []
#for i in v:
    #2.2.1: create novel data set where variable only takes on that value
#    cte_X_cp = np.copy(test_pred)
#    cte_X_cp[:,ind] = i
    #2.2.2 predict response
#    yhat = new_model.predict(cte_X_cp)
    #2.2.3 mean
#    means.append(np.mean(yhat))

#plt.plot(v, means)
#plt.show()



stop = time.time()
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')


print(f"Training time: {stop - start}s")
print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
tracemalloc.stop()



model.evaluate(test_pred,response)


plt.plot(history_loss)
plt.show()

plt.plot(mse_mean_history)
plt.show()

plt.plot(r2_history)
plt.show()



y_pred_test = model.predict(test_pred)
new_model = tf.keras.models.load_model('my_model.h5')
batch_size = 3000
y_pred_test = np.concatenate([model.predict(test_pred[i:i+batch_size]) for i in range(0, len(test_pred), batch_size)])
SS_res = 0
SS_tot = 0
for i in range(0, len(response), batch_size):
    y_test_batch = response[i:i+batch_size]
    y_pred_batch = y_pred[i:i+batch_size]
    SS_res += np.sum((y_test_batch - y_pred_batch)**2)
    SS_tot += np.sum((y_test_batch - np.mean(response))**2)
R2 = 1 - SS_res/SS_tot
print(f"R-squared: {R2}")


model.summary()

model.save('my_model_2.h5')


##VARIABLE IMPORTANCE TESTING --- UNUSED AFTER TESTING
##rf = RandomForestRegressor(n_estimators=100, random_state=42)
##rf.fit(test_pred, response)
#importances = rf.feature_importances_
#std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
#indices = np.argsort(importances)[::-1]
#
#print("Feature ranking:")
#for f in range(test_pred.shape[1]):
#    print("%d. %s (%f)" % (f + 1, test_pred_test.columns[indices[f]], importances[indices[f]]))