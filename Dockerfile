FROM python:3.8

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT main:app