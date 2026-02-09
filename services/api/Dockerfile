FROM python:3.12

WORKDIR /opt/app

COPY requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"]
RUN ["python", "-m", "spacy", "download", "pt_core_news_lg"]

COPY . .

CMD uvicorn main:app --port=8000 --host=0.0.0.0