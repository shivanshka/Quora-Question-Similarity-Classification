FROM python:3.8

WORKDIR /app
COPY . /app
#COPY ./requirements.txt /app/requirements.txt
#COPY ./setup.py /app/setup.py
#COPY ./README.md /app/README.md
#COPY ./src /app/src

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#COPY ./main.py /app/main.py

EXPOSE 8000
CMD ["python", "main.py"]