FROM python:3
WORKDIR /pythonapp
COPY requirements.txt ./
COPY pythonapp/first_program.py ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python3", "./first_program.py" ]
