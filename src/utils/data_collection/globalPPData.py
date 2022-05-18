
import pandas as pd
import hashlib

def getAllLocations(pathCsv):
    colList = ["city_ascii",'lat','lng']
    df = pd.read_csv(pathCsv,usecols=colList,sep=";",decimal='.')
    df.sample(frac=1).reset_index(drop=True)
    df = df.head(5000)
    df.rename(columns = {'city_ascii':'id','lat':'latitude','lng':'longitude'}, inplace = True)
    df["label"] = 2
    df["quality"] = 6
    df["max_mean-bright"] = None
    df["max_std_bright"] = None
    df["date"] = "2022-03-14"
    df["region"] = df["id"]

    for num in range(len(df["id"])):
        df["latitude"][num] = df["latitude"][num].replace(',','.')
        df["longitude"][num] = df["longitude"][num].replace(',','.')
        df["id"][num] = int(hashlib.sha1(df["id"][num].encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    
    df.to_csv('Data/data_science/CSV/raw-Data/Cities-not-Turbines.csv', index=False)
    print(df)
getAllLocations("Data/data_science/CSV/raw-Data/worldCities.csv")