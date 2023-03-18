FROM python:3.8

WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./main.py ./

EXPOSE 5000
CMD ["python", "main.py"]