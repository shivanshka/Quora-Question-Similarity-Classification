FROM python:3.8

WORKDIR /usr/src/app

COPY ./requirements.txt ./
COPY ./setup.py ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./main.py ./

EXPOSE 5000
CMD ["python", "main.py"]