from matplotlib import pyplot as plt
import numpy as np
import requests
import json
import cv2
import time
import re
# -----------------------------------------------------------
# all functions to get the Wind Turbine Locations with Overpass-Api
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

def getLocationsForBundesland(Bundesland):
    overpass_url = "http://overpass-api.de/api/interpreter"


    overpass_query = """
    [out:json];
    area["name"="{0}"]->.boundaryarea;
    (node["power"="generator"]["generator:source"="wind"]["manufacturer"="Enercon"] (area.boundaryarea);
    way["power"="generator"]["generator:source"="wind"]["manufacturer"="Enercon"](area.boundaryarea);
    relation["power"="generator"]["generator:source"="wind"]["manufacturer"="Enercon"](area.boundaryarea);
    );
    out center;
    """
    time.sleep(20)
    sentence = overpass_query.format(Bundesland)    
    response = requests.get(overpass_url, 
                            params={'data':sentence })
    data = response.json()
    print(f'Bundesland: {Bundesland}: {len(data["elements"])}')
    return data

def loop_through_Bund(path):
    Bundesländer = ["Hamburg", "Berlin", "Brandenburg", "Nordrhein-Westfalen",
    "Mecklenburg-Vorpommern", "Hessen", "Rheinland-Pfalz",
    "Thüringen", "Saarland", "Bayern", "Baden-Württemberg"]

    for Bundesland in Bundesländer:
        jsonData = {}
        data = getLocationsForBundesland(Bundesland)
        for element in data['elements']:
            if element['type'] == 'node':
                jsonData[element["id"]] = {"longitude": element["lon"], "latitude":element["lat"]}
            elif 'center' in element:
                jsonData[element["id"]] = {"longitude": element['center']['lon'], "latitude":element['center']['lat'] }

        #with open('/UBA_WindTurbineProject/JsonFiles/Shape_WindTurbines_LocationExport_OSM.json', 'w') as f:
        #  f.write(str(json))
        json_string = json.dumps(jsonData)
        with open(f'{path}/{Bundesland}.json', 'w') as f:
            f.write(json_string)
def get_Locations_of_Country(path):
    overpass_url = "http://overpass-api.de/api/interpreter"
    land = "Australia"
    height = 100
    overpass_query = """
    [out:json];
    area["ISO3166-1"="AU"][admin_level=2];
    (node["power"="generator"]["generator:source"="wind"]["manufacturer"](area);
    way["power"="generator"]["generator:source"="wind"]["manufacturer"](area);
    relation["power"="generator"]["generator:source"="wind"]["manufacturer"](area);
    );
    out center;
    """
    time.sleep(20)
    sentence = overpass_query.format(land)
    response = requests.get(overpass_url, params={'data':sentence })
    data = response.json()
    print(f'Bundesland: {land}: {len(data["elements"])}')
    jsonData = {}
    coords = []

    higherhundret = 0
    lower = 0
    for element in data['elements']:
            # if(len(re.findall(r'\b\d+\b',element["tags"]["height:hub"]))<1):
            #     continue
            # if( int(re.findall(r'\b\d+\b',element["tags"]["height:hub"])[0])>=100):
            #     higherhundret +=1
            # else:
            #     lower +=1
            # if element['type'] == 'node':
            #     jsonData[element["id"]] = {"longitude": element["lon"], "latitude":element["lat"], "heightInM":int(re.findall(r'\b\d+\b',element["tags"]["height:hub"])[0])}
            # elif 'center' in element:
            #     jsonData[element["id"]] = {"longitude": element['center']['lon'], "latitude":element['center']['lat'], "heightInM":int(re.findall(r'\b\d+\b',element["tags"]["height:hub"])[0]) }

            if element['type'] == 'node':
                jsonData[element["id"]] = {"longitude": element["lon"], "latitude":element["lat"]}
                lon = element['lon']
                lat = element['lat']
                coords.append((lon, lat))
            elif 'center' in element:
                jsonData[element["id"]] = {"longitude": element['center']['lon'], "latitude":element['center']['lat']}
                lon = element['center']['lon']
                lat = element['center']['lat']
                coords.append((lon, lat))

        #with open('/UBA_WindTurbineProject/JsonFiles/Shape_WindTurbines_LocationExport_OSM.json', 'w') as f:
        #  f.write(str(json))
    json_string = json.dumps(jsonData)
    with open(f'{path}/{land}-WithHeights.json', 'w') as f:
        f.write(json_string)
    X = np.array(coords)
    print(f"Amount of Wind-Turbines {X.size}")
    plt.plot(X[:, 0], X[:, 1], 'o')
    plt.title('Wind-Turbines in Germany')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.show()
    #print(f'Above 100 Meters are {higherhundret} and lower are {lower}')
path = "Data/locations"
#loop_through_Bund(path)
get_Locations_of_Country(path)