import sys
#from data_information import createSubCsv, splitTrainTest
sys.path.append("src")

from utils.data_preperation.data_information import createSubCsv


path_data = 'Data'

turbine_data = path_data +"/data_science/CSV/raw-Data/Sentinel-2-WindTurbineData.csv"
path_jpg = f'{path_data}/data_science/img_database'
path_unspinned = f'{path_data}/data_science/not_spinning_images'
# spinningCsv = path_data + "Data/data_science/CSV/spinning.csv"
csvPath = "Data/data_science/CSV/emptyMeFolder/"
UndetectedCsv = "Data/data_science/CSV/raw-Data/Cities-not-Turbines.csv"
path_Data ="Data/data_science/CSV/NewUndetectedComparison/"
def splitTrainingTest(path_labeled_csv,path_image, path_Data,csvPath):
    #SET LAST PATH TO NONE::CHANGE INTO WANTED FOLDER
    #unspin_turbines(path_labeled_csv,path_image,None)
    size = 2000
    #for all 3 sub csvs --> spin, no spin, undetectable
    # spinning,spinningTest, undetected,undetectedTest = createSubCsv(path_labeled_csv,UndetectedCsv,csvPath,size)
    createSubCsv(path_labeled_csv,UndetectedCsv,path_Data, size)
    #split into training and testing
    pathTrain = path_Data+"train-"+str(size)+".csv"
    pathTest = path_Data+"test-"+str(size)+".csv"
    # splitTrainTest(spinning,spinningTest, undetected,undetectedTest,pathTrain,pathTest,size)
    print("Here")

splitTrainingTest(turbine_data,path_jpg,path_Data,csvPath)