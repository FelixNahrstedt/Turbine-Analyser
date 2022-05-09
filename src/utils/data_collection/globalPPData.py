
import pandas as pd


def getAllWindTurbines(pathCsv):
    df = pd.read_csv(pathCsv)
    df = df[(df['primary_fuel'] == "Wind" )]
    df = df[df["capacity_mw"]<10]
    df = df[df["capacity_mw"]>1]
    print(df[["latitude","longitude"]])
getAllWindTurbines("Data/data_science/CSV/global_power_plant_database.csv")