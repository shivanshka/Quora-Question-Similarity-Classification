FROM python:3.8

WORKDIR /app

COPY ./requirements.txt ./
COPY ./setup.py ./
COPY ./README.md ./
ADD ./src ./src

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./main.py ./

EXPOSE 5000
CMD ["python", "main.py"]