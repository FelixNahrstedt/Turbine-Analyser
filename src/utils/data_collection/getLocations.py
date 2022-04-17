import requests
import json
import cv2
import time

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

path = "C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data/locations"
loop_through_Bund(path)