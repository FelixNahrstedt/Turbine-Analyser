from utils.data_information import display_data, unspin_turbines


path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
turbine_data = path_data +"/data_science/CSV/Sentinel-2-WindTurbineData.csv"
path_jpg = f'{path_data}/data_science/img_database'
path_unspinned = f'{path_data}/data_science/not_spinning_images'

path_gif = f'{path_data}/Satellite/GIF'


#display_data(turbine_data)
# go through the good quality Data and convert it into still wind turbines

unspin_turbines(turbine_data,path_jpg,path_gif,path_unspinned)