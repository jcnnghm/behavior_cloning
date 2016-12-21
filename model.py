from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cross_validation import train_test_split
from keras.regularizers import l2

nb_filters = 16
image_size = (160, 320)
input_shape = (image_size[0], image_size[1], 3)
batch_size = 32


def load_img(image_path):
    image_path = image_path.replace("/home/jcnnghm/Projects/car/behavior_cloning/", "")
    img = Image.open(image_path)
    return img

def convert_img(img):
    return (np.asarray(img.resize((image_size[1], image_size[0]))) - 128.0) / 255.0

def unconvert_img(image_mat):
    return Image.fromarray(((image_mat * 255) + 128).round().astype('uint8'))

def batch_generator_factory(input_data):
    size = (len(input_data) // batch_size) * batch_size

    def batch_generator():
        batch = [[], []]
        while True:
            np.random.shuffle(input_data)
            for image_path, angle in input_data:
                batch[0].append(convert_img(load_img(image_path)))
                batch[1].append(angle)
                if len(batch[0]) == batch_size:
                    yield np.asarray(batch[0]), np.asarray(batch[1])
                    batch = [[], []]

    return size, batch_generator()

def flip_generator_factory(batch_generator):
    size, generator = batch_generator

    def batch_generator():
        for images, labels in generator:
            yield images, labels
            batch = [[], []]
            for image_mat, angle in zip(images, labels):
                image = unconvert_img(image_mat)
                flipped_img = image.transpose(Image.FLIP_LEFT_RIGHT)
                batch[0].append(convert_img(flipped_img))
                batch[1].append(-angle)
            yield np.asarray(batch[0]), np.asarray(batch[1])

    return size * 2, batch_generator()


if __name__ == '__main__':
    df = pd.read_csv(
        'data/driving_log.csv',
        header=None,
        names=['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break_val', 'speed']
    )
    data = [(row.center_img, row.steering_angle) for row in df.itertuples()]
    train, test = train_test_split(data, test_size=0.2)
    train, validation = train_test_split(train, test_size=0.25)

    model = Sequential([
            Convolution2D(nb_filters, 3, 3, input_shape=input_shape),
            Dropout(0.2),
            ELU(),
            MaxPooling2D(pool_size=(2,2)), # 160 x 80
            Convolution2D(nb_filters * 2, 3, 3),
            Dropout(0.2),
            ELU(),
            MaxPooling2D(pool_size=(2,2)), # 80 x 40
            Convolution2D(nb_filters * 4, 3, 3),
            Dropout(0.2),
            ELU(),
            MaxPooling2D(pool_size=(2,2)), # 40 x 20
            Convolution2D(nb_filters * 8, 3, 3),
            Dropout(0.2),
            ELU(),
            MaxPooling2D(pool_size=(2,2)), # 20 x 10 x 256
            Flatten(),
            Dense(128, name='hidden1'),
            Dropout(0.5),
            ELU(),
            Dense(1, name='output'),
    ])

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mean_squared_error']
    )

    train_size, train_generator = flip_generator_factory(batch_generator_factory(train))
    validation_size, validation_generator = batch_generator_factory(validation)

    model.fit_generator(
        generator=train_generator,
        samples_per_epoch=train_size,
        nb_epoch=100,
        validation_data=validation_generator,
        nb_val_samples=validation_size
    )

    with open('model.json', 'w') as f:
        f.write(model.to_json())

    model.save_weights('model.h5')
