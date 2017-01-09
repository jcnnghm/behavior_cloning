import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from model import convert_img, preprocess_img, flip_img
import pygame


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class ManualControl(object):
    def __init__(self):
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def get_events(self):
        for event in pygame.event.get():
            yield event

    def process_events(self):
        for event in self.get_events():
            pass

    def get_throttle(self):
        # Full throttle when holding x
        if self.joystick.get_button(2):
            return 1.0

        return ((self.joystick.get_axis(5) + 1.0) / 2.0) / 5.0

    def get_angle(self):
        return self.joystick.get_axis(0) / 2.0

    def is_manual_control(self):
        return self.joystick.get_button(0)

control = ManualControl()
train_images = []
train_labels = []


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    preprocessed_img = preprocess_img(image)
    image_array = convert_img(preprocessed_img)
    transformed_image_array = image_array[None, :, :, :]

    for event in control.get_events():
        if event.type == pygame.JOYBUTTONUP and event.button == 1:
            print("Saving fine-tuned model")
            model.save_weights('model.h5')

    throttle = control.get_throttle()
    if throttle == 0.0:
        throttle = 0.2

    if control.is_manual_control():
        steering_angle = control.get_angle()

        train_images.append(image_array)
        train_labels.append(steering_angle)

        flipped_img, flipped_angle = flip_img(preprocessed_img, steering_angle)
        train_images.append(convert_img(flipped_img))
        train_labels.append(flipped_angle)
    else:
        # This model currently assumes that the features of the model are just the images. Feel free to change this.
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))
        # The driving model currently just outputs a constant throttle. Feel free to edit this.
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)

    if len(train_images) >= 64:
        print("Training on image batch")
        model.train_on_batch(np.asarray(train_images), np.asarray(train_labels))
        train_images[:] = []
        train_labels[:] = []


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__(),
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
