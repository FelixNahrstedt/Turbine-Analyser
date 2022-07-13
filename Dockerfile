FROM python:3.8

ADD requirements.txt /

WORKDIR /src

RUN apt-get update && apt-get install --yes libgdal-dev

RUN python -m pip install --upgrade pip

RUN pip install -r /requirements.txt

COPY . /src

ENTRYPOINT [ "python" ]

RUN chmod 644 app.py

CMD ["app.py"]
