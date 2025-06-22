FROM python:3.13-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
