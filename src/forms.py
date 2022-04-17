from flask_wtf import FlaskForm
from wtforms import (DecimalField, TextAreaField, IntegerField, BooleanField,
                     RadioField,DateField)
from wtforms.validators import InputRequired, Length
from wtforms.validators import NumberRange


class LatLongForm(FlaskForm):
    latitude = DecimalField('latitude',render_kw={'class':'form-control'}, validators=[InputRequired(),
                                             NumberRange(min=-90, max=90, message='The numbers are in decimal degrees format and range from -90 to 90.')])
    longitude = DecimalField('longitude',render_kw={'class':'form-control'}, validators=[InputRequired(),
                                             NumberRange(min=-180, max=180, message='The numbers are in decimal degrees format and range from -180 to 180.')])
    datumField = DateField('Datum',render_kw={'class':'form-control'}, validators=[InputRequired()])