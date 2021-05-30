import requests
import pprint
from collections import defaultdict
import calendar
from datetime import datetime
from datetime import date 
import json


def most_frequent(List): 
    return max(set(List), key = List.count) 


def _get_weather(city="",country="",js=[],lon=0, lat=0):
    url = "https://community-open-weather-map.p.rapidapi.com/forecast"

    if(lon!= 0 and lat!= 0):
        querystring = {'lat': lat, 'lon': lon, "units":"metric"}
    else:
        querystring = {"q":f"{city},{country}","units":"metric"}

    headers = {
    'x-rapidapi-host': "community-open-weather-map.p.rapidapi.com",
    'x-rapidapi-key': "f65983498fmsh7aeb620012c066ap1a8b06jsn5c9030b8fc6f"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)


    if("400 - Bad Request" not in response.text):
        resp = eval(response.text.replace("false","False").replace("true","True").replace("null","None"))
    else:
        return [], []
    # print(resp)
    
    dic_resp = defaultdict(lambda: defaultdict(list))
    for elm in resp['list']:
        date_time_obj = datetime.strptime(elm['dt_txt'], '%Y-%m-%d %H:%M:%S')
        dic_resp[calendar.day_name[date_time_obj.weekday()]]["min"].append(elm['main']['temp_min'])
        dic_resp[calendar.day_name[date_time_obj.weekday()]]["max"].append(elm['main']['temp_max'])
        dic_resp[calendar.day_name[date_time_obj.weekday()]]["wea"].append(elm['weather'][0]['main'])

    # print(dic_resp)
    
    for day, dic in dic_resp.items():
        di = {
           "City":city,
           "Day":day,
           "Weather":most_frequent(dic["wea"]),
           "Max":max(dic["max"]),
           "Min":min(dic["min"])
           }
        js.append(di)


js = []
city = ["Hong Kong","Beijing","San Franscisco","New York","Rome","London"]
country_id = ["HK","CN-HI","USA","USA","IT","UK"]
for city_, country_id_ in zip(city,country_id):
    print(city_)
    print(country_id_)
    _get_weather(city=city_,country=country_id_,js=js)

with open('weather.json', 'w') as outfile:
    json.dump(js, outfile,indent=4)

