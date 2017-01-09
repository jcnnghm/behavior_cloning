FROM jcnnghm/ml-py3:latest
MAINTAINER justin@bulletprooftiger.com

RUN /root/anaconda3/envs/venv/bin/pip install python-socketio eventlet pygame
