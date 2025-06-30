FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y git libgomp1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/bootstrap.py scripts/bootstrap.py
RUN python scripts/bootstrap.py

COPY . .

RUN pip install .

ENV FLASK_APP=app/main.py

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.main:app"]