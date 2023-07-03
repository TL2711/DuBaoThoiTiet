import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# load data
df = pd.read_csv('data/test.csv')

X = df.iloc[:, 0:10].values
label  = df.iloc[:, 13:18].values

print(X.shape)
print(label.shape)

X = np.array(X)
label = np.array(label)

print(X[0])
print(label[0])


# normalize data

x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)

y_mean = np.mean(label, axis=0)
y_std = np.std(label, axis=0)

print(x_mean)
print(x_std)

print(y_mean)
print(y_std)

X = (X - x_mean)/x_std

label = (label - y_mean) / y_std

# get 5 hours of data for each training sample and flatten
X = [X[i:i+5].ravel() for i in range(len(X)-4)]
# print(X[0])

# remove first 4 samples of label
label = label[4:, :]

# print(label[0])

X = np.array(X)
label = np.array(label)



# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2,random_state=0)


# reshape input to fit LSTM model
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# created generator to get 5 hours of data for each training sample
# def generator(X,y, step):
#   while True:
#     for i in range(0, len(X), step):
#       yield X[i:i+step].ravel(), y[i:i+step]



# build model
model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(64, input_shape=(X_train_lstm.shape[1], 1), return_sequences=True, activation='tanh'),
  tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh'),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.LSTM(128, activation='tanh'),
  tf.keras.layers.Dropout(0.2),


  tf.keras.layers.Dense(256, activation="linear"),
  tf.keras.layers.Dense(128, activation="linear"),
  tf.keras.layers.Dense(128, activation="linear"),
  tf.keras.layers.Dense(5, activation="linear")
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# train model

history = model.fit(X_train_lstm, y_train, epochs=70, batch_size= 128, validation_split=0.2)

# save model
model.save('Model/model_temp.h5')

# predict on test data
y_pred = model.predict(X_test_lstm)

# print some predictions
for i in range(10):
  print('Actual: ', y_test[i])
  print('Predicted: ', y_pred[i])
  print('------------------')



loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss for Temperature hour')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.show()

# plot accuracy

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Accuracy for Temperature hour')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# print metrics

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('R2 Score: ', r2_score(y_test, y_pred))

