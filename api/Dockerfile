FROM python:3.8-slim-buster

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5001

ENTRYPOINT [ "python" ]

CMD [ "iris_model_api.py" ]