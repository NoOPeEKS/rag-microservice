FROM python:3.10

WORKDIR /

# Create config directory
RUN mkdir /config

# Copy requirements.txt files
COPY ./config/requirements.txt /config/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /config/requirements.txt

# Copy settings.yaml file
COPY ./config/settings.yaml /config/settings.yaml
COPY ./config/main.yaml /config/main.yaml

# Copy API code
COPY ./app /app

# Copy source code
COPY ./src/ /src

# ENTRYPOINT
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
