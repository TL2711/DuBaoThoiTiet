import csv
import requests


# api_key = "7c0a239224e7429daddafb9b9de536c5"
api_key = "5e27f851fd764810be5b08f89ab34e3e"
# api_key ="b57f5d5ab486430db1d389157fd6ee1e"

city_name = "Hanoi"


start_date = "2004-05-01"
step = "604800"# 7 days
end_date = "2005-05-01"


url = f"https://api.weatherbit.io/v2.0/history/daily?city=Hanoi&country=VN&start_date={start_date}&end_date={end_date}&key={api_key}"

# Make a request to the API and get the response
response = requests.get(url)
print(response)
# Parse the response JSON
data = response.json()
print(data)

# print(data)

# Open a new file in write mode and create a csv writer object
with open('data/weather_date.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    # # Write the header row
    
    """                 "rh":70.2,
                   "wind_spd":3.8,
                   "slp":1022,
                   "max_wind_spd":6.7,
                   "max_wind_dir":220,
                   "max_wind_spd_ts":1483232400,
                   "wind_gust_spd":12.7,
                   "min_temp_ts":1483272000,
                   "max_temp_ts":1483308000,
                   "dewpt":1.8,
                   "snow":0,
                   "snow_depth":1.0,
                   "precip":10.5,
                   "precip_gpm":13.5,
                   "wind_dir":189,
                   "max_dhi":736.3,
                   "dhi":88,
                   "max_temp":10,
                   "pres":1006.4,
                   "max_uv":5,
                   "t_dhi":2023.6,
                   "datetime":"2023-04-29",
                   "temp":7.86,
                   "min_temp":5,
                   "clouds":43,
                   "ts":1483228800,
                   "revision_status":"final" """

    #writer.writerow([ 'Datetime','Date', 'Temperature', 'Humidity', 'Temp_min', 'Temp_max', 'Pressure', 'Wind_speed', 'Clouds','Precip'])
    for item in data['data']:
        # Extract the date, temperature, and humidity from the JSON data
        date = item['ts']
        wind_speed = item['wind_spd']
        humidity = item['rh']
        temp_min = item['min_temp']
        temp_max = item['max_temp']
        pressure = item['pres']
        temp = item['temp']
        clouds = item['clouds']
        datetime = item['datetime']
        precip = item['precip']

        # Write the data rows
        writer.writerow([datetime,date, temp, humidity, temp_min, temp_max, pressure, wind_speed, clouds,precip])


print("Weather data saved to weather_data.csv")



