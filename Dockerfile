#Dockerfile, Image, Container
FROM python:3.8

COPY requirements.txt ./
RUN pip install -U pip requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ADD main.py .

CMD [ "python", "./main.py"]