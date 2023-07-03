import tensorflow as tf
import numpy as np
import warnings
import GetCurrent
import GetDay

warnings.filterwarnings("ignore")


x_mean = np.array([7.74973094e+01, 2.80202640e+00, 1.42257474e+02, 2.46415703e-01,
 1.00962442e+03 ,3.00062242e+02, 6.81242020e+01 ,1.14995508e+01,
 2.97592885e+02, 4.93261710e-01])



x_std = np.array([ 15.88368275  , 1.40884157 ,101.64481521   ,0.99006853 ,  7.03036941,
   8.3542265 ,  27.69297933 ,  6.92203511 ,  5.75967504  , 0.64748694])

y_mean = np.array([297.59290423, 297.59294206, 297.59298934, 297.59301015, 297.59303474])

y_std = np.array([5.75965043, 5.75960779 ,5.75955858, 5.75953188, 5.75950288])



input = np.array(GetCurrent.input_data)

if len(input) >5:
    # remove first sample if time is XX:00
    input = input[1:]


for i in range(9):
    # check if input nan 
    if np.isnan(input[0][i]):
        input[0][i] = x_mean[i]

def predict_temp():
    model = tf.keras.models.load_model('Model/model_temp.h5')
   
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = y_pred*y_std + y_mean - 273.15
    y_pred = np.round(y_pred,1, out=None)
    return y_pred[0]



def get_icon_url(weather):
    if weather == 2:
        return 'url(:/Img/Image/icons8-rain-48.png);'
    elif weather == 0:
        return 'url(:/Img/Image/icons8-partly-cloudy-day-48.png);'
    elif weather == 1:
        return 'url(:/Img/Image/icons8-sun-48.png);'
    
def predict_weather_1h():
    model = tf.keras.models.load_model('Model/predict_weather1.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)



def predict_weather_2h():
    model = tf.keras.models.load_model('Model/predict_weather2.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_3h():
    model = tf.keras.models.load_model('Model/predict_weather3.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_4h():
    model = tf.keras.models.load_model('Model/predict_weather4.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_5h():
    model = tf.keras.models.load_model('Model/predict_weather5.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)



x_day_mean = np.array([  24.268578,    78.097824 ,   21.17552 ,    27.930298 , 1009.84357 ,  2.8037562 ,  74.292755  ,   4.3086853  ,  6.44165  ,    1.0352113])

x_day_std = np.array([   5.2879195 , 10.518557  ,  5.0207067  , 6.0099254 ,  7.1144595  , 0.86248624, 27.60367 ,   10.405915  ,  3.4642637  , 0.8310101 ])

y_day_mean = np.array([27.930317, 21.176   ])

y_day_std = np.array([6.0099273 ,5.0206585])


input_day = np.array(GetDay.input_data)

for i in range(9):
    # check if input nan 
    if np.isnan(input_day[0][i]):
        input_day[0][i] = x_day_mean[i]

def text_tomo(y):
    if y == 2:
        return 'Mưa'
    elif y == 0:
        return 'Nhiều mây'
    elif y == 1:
        return 'Nắng'

def predict_weather_day():
    model = tf.keras.models.load_model('Model/model_temp_day.h5')
   
    X = (input_day- x_day_mean)/x_day_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    result = [text_tomo(y_pred),get_icon_url(y_pred)]
    return result

def predict_temp_day():
    model = tf.keras.models.load_model('Model/model_temp_day.h5')
    X = (input_day- x_day_mean)/x_day_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = y_pred*y_day_std + y_day_mean
    y_pred = np.round(y_pred,1, out=None)
    result = str(round(y_pred[0][0])) + '/' +  str(round(y_pred[0][1]))
    return result


