import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('data/weather_date.csv')
df = df.dropna()

for i in range(len(df['weather'])):
    if df['weather'][i] == 'rain':
        df['weather'][i] = 2
    elif df['weather'][i] == 'sunny':
        df['weather'][i] = 1
    elif df['weather'][i] == 'clouds':
        df['weather'][i] = 0

X = df.iloc[:, 2:12].values
y = df.iloc[:, -1].values

print(X[0])
print(y[0])


X = X[:-1, :]
y = y[:-1]



print(X[0])
print(y[0])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# normalize data stardardization

x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)

print(x_mean)
print(x_std)

X = (X - x_mean) / x_std

def one_hot_encode(y):
    one_hot = np.zeros((len(y), 3))
    for i in range(len(y)):
        one_hot[i, int(y[i])] = 1
    return one_hot

y = one_hot_encode(y)

X = [X[i:i+5].ravel() for i in range(len(X)-4)]

# remove first 4 samples of label
y = y[4:]

X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

# reshape data to 3D

X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_train_lstm.shape)
print(X_test_lstm.shape)

# build model
model = tf.keras.Sequential([
  tf.keras.layers.LSTM(64, input_shape=(X_train_lstm.shape[1], 1), return_sequences=True, activation='tanh'),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.LSTM(64, activation='tanh'),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train_lstm, y_train,  epochs=50,  batch_size= 32 , validation_data=(X_test_lstm, y_test))

# plot loss and accuracy

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss of weather classifier day')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy of weather classifier day')
plt.legend()
plt.show()

# predict test data

y_pred = model.predict(X_test_lstm)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred))

# plot confusion matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix of weather classifier day')
plt.show()


# save model

model.save('model/weather_day.h5')




