from flask_wtf import FlaskForm
import pandas as pd
from wtforms import (DecimalField, TextAreaField, IntegerField, BooleanField,
                     RadioField,DateField)
import matplotlib.pyplot as plt
from wtforms.validators import InputRequired, Length
from wtforms.validators import NumberRange

class LatLongForm(FlaskForm):
    latitude = DecimalField('latitude',render_kw={'class':'form-control'}, validators=[InputRequired(),
                                             NumberRange(min=-90, max=90, message='The numbers are in decimal degrees format and range from -90 to 90.')])
    longitude = DecimalField('longitude',render_kw={'class':'form-control'}, validators=[InputRequired(),
                                             NumberRange(min=-180, max=180, message='The numbers are in decimal degrees format and range from -180 to 180.')])
    datumField = DateField('Datum',render_kw={'class':'form-control'}, validators=[InputRequired()])


def plotForThesis():
    path = "Data/data_science/CSV/baseModel.csv"
    df = pd.read_csv(path)
    lines = df.plot.line(x='epoch', y=['train_acc', 'val_acc', "train_loss","val_loss"])
    plt.title('Untuned Loss and Accuracy of Turbine Base Model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['training accuracy', 'validation accuracy',"training loss", "validation loss"], loc='lower right')
    plt.savefig("Data/data_science/PLOTS/overfit-BaseModel_not_adapted-lr-0-005.png")

    plt.show()
plotForThesis()