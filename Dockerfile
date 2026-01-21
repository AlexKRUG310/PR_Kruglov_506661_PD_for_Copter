FROM python:3.12-slim-bullseye
WORKDIR /PD_for_copter
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /PD_for_copter/src
CMD ["python","main.py"]