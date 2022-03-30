# Sentinel 2 Bewegungserkennung
# Windows
Activate Virtual Environment: .\venv\Scripts\activate

# How to Use:

1. install packages from requirements txt
2. install the 4 packages from the .whl files (if not working, you need to install Fiona, GDAL, pyhdf and rasterio for your system) 
--> For .whl for other systems: 
https://www.lfd.uci.edu/~gohlke/pythonlibs/
3. either use functionality or directly start main file

# Csv Label Database:

Name-Bild: ID-Datum-Band.jpg
ID: Bild-Id
lat: latitude
lon: longitude

Label-Bild: 
            0 = Drehendes Windrad, 
            1 = Stehendes Windrad, 
            2 = Nicht erkennbar

Erkennbarkeit: 
            1 = 3 Rotoren Bewegen sich klar, 
            2 = 1-2 Rotoren Bewegen sich klar, 
            3 = Bewegung erkennbar, 
            4 = Bewegung erkennbar wenn man weiß wo das Windrad steht, 
            5 = Bewegung wahrscheinlich, wenn man weiß wo windrad steht

Mean-Bild-hell: durchschnittshelligkeit des hellsten bandes
std-Bild-hell: standardabweichung des hellsten bildes
Datum: Datum 
Region: Bundesland