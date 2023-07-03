import requests
import datetime
import csv
import time

api_key = "ccc9d9071252498cbd589c8689961bd8"
city = "Hanoi,VN"

start_date = datetime.date(2019, 12, 31)
end_date = start_date + datetime.timedelta(days=13)



while end_date <= datetime.date(2023, 1, 1):
    # create url with start and end date
    url = f"https://api.weatherbit.io/v2.0/history/hourly?&city={city}&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&tz=local&units=S&key={api_key}"
    
    # make API request and get data
    response = requests.get(url)
    print(response)
    data = response.json()["data"]
    print(data)
    # write data to csv
    with open('data/data_hour.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            humidity = item["rh"]
            temperature = item["temp"]
            wind_speed = item["wind_spd"]
            wind_dir = item["wind_dir"]
            precipitation = item["precip"]
            pressure = item["pres"]
            app_temp = item["app_temp"]
            cloud_cover = item["clouds"]
            timestamp = item["timestamp_local"]
            weather_description = item["weather"]["description"]
            
            writer.writerow([humidity, temperature, wind_speed, wind_dir, precipitation, pressure, app_temp, cloud_cover, timestamp, weather_description])
    
    # update dates for next iteration
    start_date = end_date + datetime.timedelta(days=1)
    end_date = start_date + datetime.timedelta(days=13)
    time.sleep(40)