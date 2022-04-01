# Sentinel 2 Bewegungserkennung
# Windows
Activate Virtual Environment: .\venv\Scripts\activate

# How to Use:

1. install packages from requirements txt
2. install the 4 packages from the .whl files (if not working, you need to install Fiona, GDAL, pyhdf and rasterio for your system) 
--> For .whl for other systems: 
https://www.lfd.uci.edu/~gohlke/pythonlibs/
3. either use functionality or directly start main file

# image database
4.762x3 images

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
            Dreht Sich: 
            1 = 3 Rotoren Bewegen sich klar, 
            2 = 1-2 Rotoren Bewegen sich klar, 
            3 = Bewegung erkennbar, 
            4 = Bewegung erkennbar wenn man weiß wo das Windrad steht, 
            5 = Bewegung wahrscheinlich, wenn man weiß wo windrad steht
            Dreht Sich Nicht: 
            1 = 3 Rotoren stehen eindeutig, 
            2 = Keiner der Rotoren bewegt sich über ein minimales Rauschen hinaus, 
            3 = keine Bewegung erkennbar, Windrad erkennbar, 
            4 = keine Bewegung erkennbar wenn man weiß wo das Windrad steht, 
            5 = Bewegung unwahrscheinlich, wenn man weiß wo windrad steht
Mean-Bild-hell: durchschnittshelligkeit des hellsten bandes
std-Bild-hell: standardabweichung des hellsten bildes
Datum: Datum 
Region: Bundesland

# Beauty Shots: 

PlaneBow: 
52.5783847 10.914772
Bundesland = Niedersachsen; Windräder durch = 34

RGB at its Peak: 
53.4673175 7.3058046
Bundesland = Niedersachsen; Windräder durch = 210

Bildverschnitt: 
53.1892 8.272147
Bundesland = Niedersachsen; Windräder durch = 737

# Wo ist es schwer erkennbar: 
Schatten auf See/Meer:
53.4222285 10.4005054
Bundesland = Schleswig-Holstein; Windräder durch = 0 - 310

Wind steht Schief: 
54.6254457 9.0608869
Bundesland = Schleswig-Holstein; Windräder durch = 534
oder
53.5369481 7.2219562
Bundesland = Niedersachsen; Windräder durch = 1

Grenze des Satellitenbildes: 
54.2855592 8.9995802
Bundesland = Schleswig-Holstein; Windräder durch = 692

Kein Schattenwurf: (Peter Pans Windrad???)
54.08834 9.2468273
Bundesland = Schleswig-Holstein; Windräder durch = 778
oder
54.0901383 9.2411415
Bundesland = Schleswig-Holstein; Windräder durch = 779

Wolken trotz Cloud Filter:
53.5357711 7.2256459
Bundesland = Niedersachsen; Windräder durch = 5
53.5300905 7.219292
Bundesland = Niedersachsen; Windräder durch = 4
