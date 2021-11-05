FROM python:3.7.12

RUN apt-get update 
# RUN apt-get install -y python-pip
COPY ./dockerfile .
COPY ./requirements.txt .
COPY ./test.py .
RUN pip install -r requirements.txt
# boto3

ENTRYPOINT [ "python" ]
CMD [ "test.py" ]

# 아 이거 dockerfile 작성해서 docker image 만든느 거구나..^^