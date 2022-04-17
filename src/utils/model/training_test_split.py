import sys
import os
sys.path.append("C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/src")

from utils.data_preperation import data_information


path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'

turbine_data = path_data +"/data_science/CSV/Sentinel-2-WindTurbineData.csv"
path_jpg = f'{path_data}/data_science/img_database'
path_unspinned = f'{path_data}/data_science/not_spinning_images'
csvPath = path_data + "/data_science/CSV/"
spinningCsv = path_data + "/data_science/CSV/spinning.csv"
notSpinningCsv = path_data + "/data_science/CSV/not_spinning.csv"
UndetectedCsv = path_data + "/data_science/CSV/undetected.csv"
TestSet = path_data + "/data_science/CSV/TestSet.csv"
TrainSet = path_data + "/data_science/CSV/TrainSet.csv"
def splitTrainingTest(path_labeled_csv,path_image,path_splitting_Csv, path_trainSet, path_testSet):
    #SET LAST PATH TO NONE::CHANGE INTO WANTED FOLDER
    #unspin_turbines(path_labeled_csv,path_image,None)
    #for all 3 sub csvs --> spin, no spin, undetectable
    #spinningCsv,notSpinningCsv, UndetectedCsv = createSubCsv(path_labeled_csv,path_splitting_Csv)
    #split into training and testing
    #splitTrainTest(spinningCsv, notSpinningCsv,UndetectedCsv,path_trainSet,path_testSet)
    print("Here")

splitTrainingTest(turbine_data,path_jpg,csvPath,TrainSet, TestSet)