import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.cross_validation import train_test_split

nb_filters = 8
image_size = (80, 160)
crop_size = (0, 25, 160, 75)  # 160 x 50
input_shape = (crop_size[3] - crop_size[1], crop_size[2] - crop_size[0], 3)
batch_size = 256


def load_img(image_path):
    image_path = image_path.strip().replace(
        "/home/jcnnghm/Projects/car/behavior_cloning/", "/work/"
    ).replace(
        "/home/jcnnghm/Desktop/sim2/", "/work/"
    )
    img = Image.open(image_path)
    return preprocess_img(img)

def preprocess_img(img):
    img = img.resize((image_size[1], image_size[0]))
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    img = img.crop(crop_size)
    return img

def convert_img(img):
    return (np.asarray(img) - 128.0) / 255.0

def unconvert_img(image_mat):
    return Image.fromarray(((image_mat * 255) + 128).round().astype('uint8'))

def flip_img(img, angle):
    return img.transpose(Image.FLIP_LEFT_RIGHT), -angle

# These are the transformations that are supported
def plain_img(image_mat, angle):
    return image_mat, angle

def noisy_img(image_mat, angle):
    noisy_image_mat = image_mat + np.random.normal(scale=0.01, size=image_mat.shape)
    return noisy_image_mat, angle

def brightness_img(image_mat, angle):
    img = unconvert_img(image_mat)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(np.random.normal(1.0, 0.1))
    return convert_img(img), angle



def test_generator(input_rows):
    def image_picker(row):
        return convert_img(load_img(row[0])), row[1]

    input_data = [(row.center_img, row.steering_angle) for row in input_rows]
    return batch_generator(input_data, [], image_picker)

def train_generator(input_rows):
    def image_picker(row):
        img_path, angle = row
        img = load_img(img_path)
        if np.random.random() < 0.5:
            img, angle = flip_img(img, angle)

        return convert_img(img), angle

    input_data = []
    for row in input_rows:
        input_data.append((row.center_img, row.steering_angle))
        input_data.append((row.left_img, row.steering_angle + 0.25))
        input_data.append((row.right_img, row.steering_angle - 0.25))

    return batch_generator(
        input_data,
        [plain_img, noisy_img, brightness_img],
        image_picker,
        include_sample_weight=True
    )

def batch_generator(input_data, transformations, image_picker, include_sample_weight=False):
    batch = [[], [], []]
    while True:
        np.random.shuffle(input_data)
        for row in input_data:
            img, angle = image_picker(row)
            sample_weight = 5.0 * abs(angle) + 0.1

            if transformations:
                transform = np.random.choice(transformations)
                img, angle = transform(img, angle)
            batch[0].append(img)
            batch[1].append(angle)
            batch[2].append(sample_weight)
            if len(batch[0]) == batch_size:
                yield np.asarray(batch[0]), np.asarray(batch[1]), np.asarray(batch[2])
                batch = [[], [], []]


class Model(object):
    def __init__(self):
        self.model = Sequential([
                Convolution2D(nb_filters, 7, 7, input_shape=input_shape),
                Dropout(0.3),
                ELU(),
                MaxPooling2D(pool_size=(2,2)), # 80 x 25
                Convolution2D(nb_filters * 2, 3, 3),
                Dropout(0.3),
                ELU(),
                MaxPooling2D(pool_size=(2,2)), # 40 x 12
                Convolution2D(nb_filters * 4, 3, 3),
                Dropout(0.3),
                ELU(),
                MaxPooling2D(pool_size=(2,2)), # 20 x 6
                Flatten(),
                Dense(64, name='hidden1'),
                Dropout(0.5),
                ELU(),
                Dense(1, name='output'),
        ])
        self.model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['mean_squared_error']
        )
        self.test_data = []

    def save(self):
        with open('model.json', 'w') as f:
            f.write(self.model.to_json())

        self.model.save_weights('model.h5')

    def evaluate(self):
        test_mse = self.model.evaluate_generator(
            generator=test_generator(self.test_data),
            val_samples=self._samples_per_epoch(self.test_data)
        )

        print("Test MSE: {}".format(test_mse))

    def _samples_per_epoch(self, data):
        return (len(data) // batch_size) * batch_size

    def train(self, log_file, epochs, train_name='main'):
        df = pd.read_csv(
            log_file,
            header=None,
            names=['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break_val', 'speed']
        )
        data = [row for row in df.itertuples()]
        train, test = train_test_split(data, test_size=0.2)
        train, validation = train_test_split(train, test_size=0.25)
        self.test_data.extend(test)

        checkpoint = ModelCheckpoint(
            "checkpoints/{train_name}-weights.{{epoch:02d}}-{{val_loss:.4f}}.h5".format(train_name=train_name),
            save_weights_only=True
        )

        history = self.model.fit_generator(
            generator=train_generator(train),
            samples_per_epoch=self._samples_per_epoch(train),
            nb_epoch=epochs,
            validation_data=test_generator(validation),
            nb_val_samples=self._samples_per_epoch(validation),
            callbacks=[checkpoint],
            max_q_size=80,
            nb_worker=6,
            pickle_safe=True,
        )

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("{train_name}-loss.png".format(train_name=train_name))


if __name__ == '__main__':
    model = Model()
    model.train('hard_data/driving_log.csv', epochs=5, train_name='hard')
    model.train('gold_data/driving_log.csv', epochs=5, train_name='main')
    model.evaluate()
    model.save()
