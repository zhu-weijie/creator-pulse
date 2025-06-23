FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y git libgomp1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install .

ENV FLASK_APP=app/main.py

EXPOSE 8080

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]