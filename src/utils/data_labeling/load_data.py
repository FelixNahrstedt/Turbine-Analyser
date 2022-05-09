import json
import csv

import pandas as pd

# -----------------------------------------------------------
# all functions for "normal" change detection in Images
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------
csv_path = "Data/data_science/CSV/Sentinel-2-WindTurbineData.csv"
path_csv_Folder = "Data/data_science/CSV/"
def import_json(name):
    jsonData = {}
    with open(f'Data/locations/{name}.json') as f:
        jsonData = json.load(f)
    turbine_keys = []
    for keys in jsonData:
        turbine_keys.append(keys)
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
    

def small_or_big_turbines(nameJson, path_csv,path_folder, nameSmall,nameBig):
    #load Germany-WithHeights.json
    data,keys = import_json(nameJson)
    #load Sentinel-2-WindTurbineData.csv
    df = pd.read_csv(path_csv)
    # with open(path_folder+"Turbines-With-Heights.csv", 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['id','label','quality', 'date','region', "heighInM"])
    #df_by_indices = df.iloc[keys, :]
    indizes = []
    heights = []
    for key in keys:
        
        index = df.index[df["id"]==int(key)].tolist()
        if(len(index)==1):
            indizes.append(index[0])
            heights.append(data[key]["heightInM"])
        # with open(path_folder+"Turbines-With-Heights.csv", 'a', encoding='UTF8', newline='') as f:
        #     writer = csv.writer(f)
        #     # write the data
        #     writer.writerow(allData)
    df_by_indices = df.iloc[indizes, :]
    tiny = []
    small = []
    medium = []
    large = []
    xlarge = []
    df_by_indices["heightInM"] = heights
    for index, row in df_by_indices.iterrows():
        if(row["heightInM"]<73):
            tiny.append(index)
        elif(row["heightInM"]>=73 and row["heightInM"]<95):
            small.append(index)
        elif(row["heightInM"]>=95 and row["heightInM"]<120):
            medium.append(index)
        elif(row["heightInM"]>=120 and row["heightInM"]<140):
            large.append(index)
        else:
            xlarge.append(index)
    df_tiny = df.iloc[tiny,:]
    df_small = df.iloc[small,:]
    df_medium = df.iloc[medium,:]
    df_large = df.iloc[large,:]
    df_xlarge = df.iloc[xlarge,:]

    print(len(tiny),len(small),len(medium),len(large),len(xlarge))
    df_tiny.to_csv(path_csv_Folder+"Turbines-With-Heights_tiny.csv",index=False)
    df_small.to_csv(path_csv_Folder+"Turbines-With-Heights_small.csv",index=False)
    df_medium.to_csv(path_csv_Folder+"Turbines-With-Heights_medium.csv",index=False)
    df_large.to_csv(path_csv_Folder+"Turbines-With-Heights_large.csv",index=False)
    df_xlarge.to_csv(path_csv_Folder+"Turbines-With-Heights_xlarge.csv",index=False)

    #df_by_indices.to_csv(path_csv_Folder+"Turbines-With-Heights.csv",index=False)
    #go through Germany-WithHeights and for each id, check the csv for the Evaluation 

    #if(Height>100) data --> into first csv else --> data into second csv
#small_or_big_turbines("Germany-WithHeights",csv_path,path_csv_Folder,"smallTurbines","bigTurbines")