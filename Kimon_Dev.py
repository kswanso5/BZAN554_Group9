import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow.keras.backend as K


price_test = pd.read_csv('pricing_test.csv')

possibilites = [0,1,2 ,3 ,4 ,5 ,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 ]
price_test.columns = ['sku', 'price','quantity', 'order', 'duration','category']

X_testArray = []
Y_testArray = []
#brings in test record one at a time, creates dummie values, and append to X_testArray and Y_testArray
for record in pd.read_csv('pricing_test.csv', chunksize= 1):
    if record.isna().sum().sum() > 1:
        continue   #Skip over na's
    exists_test = record["category"].tolist()
    difference_test = pd.Series([item for item in possibilites if item not in exists_test])
    target_test = record["category"].append(pd.Series(difference_test))
    target_test = target_test.reset_index(drop=True)
    dummies_test = pd.get_dummies(target_test)
    dummies_test = dummies_test.drop(dummies_test.index[list(range(len(dummies_test)-len(difference_test), len(dummies_test)))])
    dummiesdf_test = pd.DataFrame(dummies_test)
    record_test = record.merge(dummiesdf_test, how='cross')  #merge dummies to record
    print(record_test)
    y_test = record_test['quantity'].values
    x_test = record_test.drop(['sku','category', 'quantity'], axis=1).values
    x_array_test = np.array(x_test)   #convert x (predictors) to array
    print(y_test)
    print(x_array_test)
    X_testArray.append(x_array_test)
    Y_testArray.append(y_test)


# Scale the data
scaler = MinMaxScaler()
X_testArray_scaled = scaler.fit_transform(np.concatenate(X_testArray))

history_loss = [] #create empty list to store loss
r2_history = [] #store r-squared scores

##Create tf model
inputs = tf.keras.layers.Input(shape=(36,1 ), name='input')  
hidden1 = tf.keras.layers.Dense(units=4, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=4, activation="sigmoid", name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

model = tf.keras.Model(inputs = inputs, outputs = output)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function(experimental_relax_shapes=True)
def train_fn(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = tf.keras.losses.mean_squared_error(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)  # calculate mean loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def predict_fn(model, x):
    x_tensor = tf.convert_to_tensor(x)
    x_tensor = tf.reshape(x_tensor, (1, -1, 1))
    return model(x_tensor)[0][0]


def r_squared(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

class RSquared(tf.keras.metrics.Metric):
    def __init__(self, name='r_squared', **kwargs):
        super(RSquared, self).__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        self.ss_res.assign_add(ss_res)
        self.ss_tot.assign_add(ss_tot)

    def reset_states(self):
        self.ss_res.assign(0.0)
        self.ss_tot.assign(0.0)

    def result(self):
        return 1 - (self.ss_res / (self.ss_tot + K.epsilon()))

history_loss = [] #create empty list to store loss
r2_history = [] #store r-squared scores

#################TRAINING###############################################
##Read in price data 1 row at a time
for record in pd.read_csv('pricing.csv', chunksize= 1):
    if record.isna().sum().sum() > 1:
        continue   #Skip over na's
    exists = record["category"].tolist()
    difference = pd.Series([item for item in possibilites if item not in exists])
    target = record["category"].append(pd.Series(difference))
    target = target.reset_index(drop=True)
    dummies = pd.get_dummies(target)
    dummies = dummies.drop(dummies.index[list(range(len(dummies)-len(difference), len(dummies)))])
    dummiesdf = pd.DataFrame(dummies)
    record = record.merge(dummiesdf, how = 'cross')  #merge dummies to record
    print(record)
    y = record['quantity'].values
    x = record.drop(['sku','category', 'quantity'], axis=1).values
    x_array = np.array(x)   #convert x (predictors) to array

    #Scaling the data
    scaler = MinMaxScaler()
    X_testArray_scaled = scaler.fit_transform(np.concatenate(X_testArray))
    X_testArray_scaled = np.expand_dims(X_testArray_scaled, axis=-1) # add extra dimension

    
    model.compile(loss = 'mse', optimizer = optimizer, metrics=[r_squared])

    loss = train_fn(model, x_array_scaled, y)
    print(loss)
    history_loss.append(loss)
    
    # Calculate R-squared score for the training data
    y_train_pred = predict_fn(model, x_array_scaled)
    r2 = RSquared(name='r_squared')
    y = tf.cast(y, dtype=tf.float32)  # Cast y to float type
    y_train_pred = tf.cast(y_train_pred, dtype=tf.float32)  # Cast y_pred to float type
    r2.update_state(y, y_train_pred)
    r2_train = r2.result().numpy()
    r2_history.append(r2_train)
    print(f'R-squared score: {r2_train}')

print(f'Training loss: {np.mean(history_loss)}')
print(f'Training R-squared: {np.mean(r2_history)}')



#################TESTING###############################################
# Calculate R-squared score for the testing data
if X_testArray and Y_testArray:
    X_testArray_scaled = scaler.transform(np.concatenate(X_testArray))
    y_pred_list = []
    for x_test in X_testArray_scaled:
        y_pred = predict_fn(model, x_test)
        y_pred_list.append(y_pred.numpy().tolist())

    y_test = np.concatenate(Y_testArray)
    r2_test = r_squared(y_test, np.concatenate(y_pred_list))
    print(f'Test R-squared: {r2_test}')

    # Calculate R-squared score for the testing data using sklearn's r2_score
    y_pred_test = np.concatenate(y_pred_list)
    r2_score_test = r2_score(y_test, y_pred_test)
    print(f"Test R-squared using sklearn's r2_score: {r2_score_test}")
else:
    print("No valid records in the test dataset after removing the NA values.")


# Calculate R-squared score for the testing data
y_test = np.concatenate(Y_testArray)
y_pred_test = np.concatenate(y_pred_list)
r2_score = r2_score(y_test, y_pred_test)
print(f'Testing R-squared: {r2_score}')

#plots loss
plt.plot(np.arange(len(history_loss)), np.array(history_loss).reshape(-1, 1))
plt.show()

# plot r-squared scores
plt.plot(np.arange(len(r2_history)), np.array(r2_history).reshape(-1, 1))
plt.title('R-squared Scores')
plt.xlabel('Epoch')
plt.ylabel('R-squared')
plt.show()


plt.scatter(np.arange(len(r2_history)), r2_history)
plt.title('R-squared scores for each training record')
plt.xlabel('Record number')
plt.ylabel('R-squared score')
plt.show()

