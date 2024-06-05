FROM python:3.12.3

RUN apt-get update

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

CMD ["bash"]

EXPOSE 8000