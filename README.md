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
4.762x3 images --> 8-bit 40x40 pixel
# left   = long + 0.00165
# right  = long - 0.00185
# bottom = lat + 0.0030
# top    = lat - 0.0005

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
oder
53.7313747 13.3508563
Bundesland = Mecklenburg-Vorpommern; Windräder durch = 106
or
53.7137076 13.3284545
Bundesland = Mecklenburg-Vorpommern; Windräder durch = 119

RGB at its Peak: 
53.4673175 7.3058046
Bundesland = Niedersachsen; Windräder durch = 210

Bildverschnitt: 
53.1892 8.272147
Bundesland = Niedersachsen; Windräder durch = 737

Car on Brandenburger Road: 
53.0409301 13.0192068
Bundesland = Brandenburg; Windräder durch = 288

Cars on Autobahn: 
2022-03-09
49.3022789 11.5465232
Bundesland = Bayern; Windräder durch = 40

Space Rainbows: 
51.0151836, 8.1369041
Bundesland = Nordrhein-Westfalen; Windräder durch = 74

fügen die Bilder zusammen???
2022-03-19
49.3065742 11.5390677
Bundesland = Bayern; Windräder durch = 41

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
oder
50.2224796 11.53903
Bundesland = Bayern; Windräder durch = 65

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

In Waldgebieten: 
52.018377, 13.1880972
Bundesland = Brandenburg; Windräder durch = 187
oder besser
51.975992 13.4212314
Bundesland = Brandenburg; Windräder durch = 644

über Solaranlagen:
52.2036865 7.6222034
Bundesland = Nordrhein-Westfalen; Windräder durch = 600

Zu große Bandunterschiede:
51.5007078 8.8463954
Bundesland = Nordrhein-Westfalen; Windräder durch = 755

Komische andere Schatten??
49.773489 8.302044
Bundesland = Rheinland-Pfalz; Windräder durch = 131

Viele Autos????? oder regenbogen ??? oder so??
49.3681291 10.8613009
Bundesland = Bayern; Windräder durch = 15

# NAMECHANGE
after changing spinning images with quality 1 or 2 into unspinning images, the new not spinning images have the name -->  ID-unspin-Datum-Band.jpg

Normales Modell: 

Test on:
49,587181, 7,639779 at 04.10.2020