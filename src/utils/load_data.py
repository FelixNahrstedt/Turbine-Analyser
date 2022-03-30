import json
import csv

# -----------------------------------------------------------
# all functions for "normal" change detection in Images
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

def import_json(Bundesland):
    jsonData = {}
    with open(f'Data/locations/{Bundesland}.json') as f:
        jsonData = json.load(f)
    turbine_keys = []
    for keys in jsonData:
        turbine_keys.append(keys)
    #deleting old Files
    # Fetch Sentinel Images
    # windpark_sw_nauen_bb = [12.7878662, 12.7878946,52.5851616,52.5851563]
    # Bounding box coordinates
    return jsonData, turbine_keys
