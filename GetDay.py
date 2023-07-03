import requests
import datetime

now = datetime.datetime.now()

now = now.strftime("%Y-%m-%d")

start = datetime.datetime.now() - datetime.timedelta(days=5)

start = start.strftime("%Y-%m-%d")


url = f'https://api.weatherbit.io/v2.0/history/daily?&city=Hanoi&country=VN&start_date={start}&&end_date={now}&key=f7f4fb4d3ad146e4862e400a8038631b'


response = requests.get(url)
data = response.json()

# get data Temperature,Humidity,Temp_min,Temp_max,Pressure,Wind_speed,Clouds,Precip,Month,weather

input_data = []


def get_weather(clound,precip):
    if precip > 0:
        return 2
    elif clound > 70:
        return 0
    
    return 1

for i in data['data']:
    humidity = i['rh']
    temp = i['temp']
    temp_min = i['min_temp']
    temp_max = i['max_temp']
    pressure = i['pres']
    wind_speed = i['wind_spd']
    clouds = i['clouds']
    precip = i['precip']
    month = int(i['datetime'][5:7])
    input_data.append([humidity,temp,temp_min,temp_max,pressure,wind_speed,clouds,precip,month,get_weather(clouds,precip)])


