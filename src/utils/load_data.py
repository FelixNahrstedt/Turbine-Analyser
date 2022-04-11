import json
import csv

from SatelliteTurbinesDataset import SatelliteTurbinesDataset

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

def load_data(path_turbine_csv, bands):
    #laod ID and Date for good Quality Data of spinning turbines (Class 1-3) + unrecognized Data
    file = open(path_turbine_csv)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    print(header)
    goodQualSpin =  []
    notRecognized = []
    notSpinning = []
    for row in rows:
        if(row[3] == '0' and (row[4] == '1' or row[4] == '2' )):
            #append date and ID for later image loading
            goodQualSpin.append([row[0],row[7]])
        if(row[3] == '2'):
            notRecognized.append(row[0],row[7])
        if(row[3] == '1' and (row[4] =='1' or  row[4] == '2'  or row[4] == '3'  )):
            notSpinning.append(row[0], row[7])
    
    #Load the Images of goodQual Spin / unrecognized --> SHape = TurbinesxCxHxW 

    batch_size = 64
    #transformed_dataset = SatelliteTurbinesDataset(ims=X_train)
    #train_dl = DataLoader(transformed_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)

