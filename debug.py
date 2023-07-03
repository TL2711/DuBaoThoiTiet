import requests
import datetime
import json

now = datetime.datetime.now()

now = now.strftime("%Y-%m-%d")

next = datetime.datetime.now() + datetime.timedelta(days=1)

next = next.strftime("%Y-%m-%d")




url = f'https://api.weatherbit.io/v2.0/history/hourly?&city=Hanoi&country=VN&start_date={now}&&end_date={next}&key=f7f4fb4d3ad146e4862e400a8038631b'


response = requests.get(url)

data = response.json()

now = datetime.datetime.now()

prev = now - datetime.timedelta(hours=5)

now = now.strftime("%Y-%m-%dT%H:%M:%S")

prev = prev.strftime("%Y-%m-%dT%H:%M:%S")

# get 5h previor from now by check timestamp_local

# get data humidity,wind_speed,wind_dir,precip,pressure,app_temp,cloud,hour,month,temp
input_data = []

weather_map = {
    0:["fog","haze","smoke","dust","mist","foggy","hazy","smoky","dusty","misty"],
    0:["cloudy","overcast clouds","clouds","cloud","partly cloudy","mostly cloudy"],
    1:["clear sky","clear","sunny","sun","broken clouds","sky","scattered clouds","few clouds","mostly sunny","partly sunny","partly sun","mostly clear"],
    2:["rain","rainy","drizzle","drizzling","shower","showers","thunderstorm","thunderstorms","thunder","storm","stormy","light rain","light showers","light drizzle","light thunderstorm","light thunderstorms","light thunder","light storm","light stormy","heavy rain","heavy showers","heavy drizzle","heavy thunderstorm","heavy thunderstorms","heavy thunder","heavy storm","heavy stormy"],

}

def get_weather(weather):
    for key, value in weather_map.items():
        if weather in value:
            return key

for  i in range(len(data['data'])):
    if data['data'][i]['timestamp_local'] < now and data['data'][i+1]['timestamp_local'] > prev :
        humidity = data['data'][i]['rh']
        wind_speed = data['data'][i]['wind_spd']
        wind_dir = data['data'][i]['wind_dir']
        precip = data['data'][i]['precip']
        pressure = data['data'][i]['pres']
        app_temp = data['data'][i]['app_temp'] + 273.15
        cloud = data['data'][i]['clouds']
        hour = datetime.datetime.strptime(data['data'][i]['timestamp_local'], '%Y-%m-%dT%H:%M:%S').hour
        temp = data['data'][i]['temp'] + 273.15
        weather = data['data'][i]['weather']['description']
        if i == 0:
            UV = data['data'][i]['uv']
            visibility = data['data'][i]['vis']
            data_now = [humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp,get_weather(weather.lower()),UV,visibility]

        input_data.append([humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp,get_weather(weather.lower())])

print(input_data)