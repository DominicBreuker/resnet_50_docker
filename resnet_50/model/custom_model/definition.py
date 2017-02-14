import os
import numpy as np
import h5py
from datetime import datetime
from keras.layers import Input, Flatten, Dense, AveragePooling2D, Convolution2D
from .methods.build_compile import Fitter, Compiler


class CustomHead(object):
    def __init__(self, num_classes=None):
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            W, b = self._load_custom_weights(False)
            self.num_classes = W.shape[1]

    def get_num_classes(self):
        return self.num_classes

    def add_layers(self, x, fully_convolutional):
        if fully_convolutional:
            return self.layers_fcn(x)
        else:
            return self.layers_normal(x)

    def layers_normal(self, x):
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(self.num_classes, activation='softmax', name='custom_fc')(x)
        return x

    def layers_fcn(self, x):
        x = Convolution2D(self.num_classes, 1, 1, border_mode='same',
                          name="custom_fc")(x)
        return x

    def build(self, input_shape):
        feature_input = Input(shape=input_shape)
        output = self.layers_normal(feature_input)
        compiler = Compiler(self)
        return compiler.compile(feature_input, output)

    def fit(self, model, X, y):
        fitter = Fitter(self)
        fitter.fit(model, X, y)

    def save(self, model):
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = "model_weights-{}.h5".format(current_time)
        model.save(os.path.join(self._weights_dir(), filename))

    def load_weights_into(self, model, fully_convolutional):
        W, b = self._load_custom_weights(fully_convolutional)
        model.get_layer("custom_fc").set_weights([W, b])

    def _load_custom_weights(self, fully_convolutional):
        f = h5py.File(self._latest_weights_file(), 'r')
        W = f['model_weights']['custom_fc']['custom_fc_W:0'].value
        b = f['model_weights']['custom_fc']['custom_fc_b:0'].value
        if fully_convolutional:
            W = self._convolutionize(W)
        return W, b

    def _convolutionize(self, W):
        return np.expand_dims(np.expand_dims(W, axis=0), axis=0)

    def _weights_dir(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "weights")

    def _latest_weights_file(self):
        weight_files = []
        for root, directories, filenames in os.walk(self._weights_dir()):
            for filename in filenames:
                if filename.endswith(".{}".format('h5')):
                    if filename.startswith('model_weights'):
                        weight_files.append(os.path.join(self._weights_dir(),
                                                         filename))
        weight_files.sort()
        if len(weight_files) == 0:
            return None
        else:
            return os.path.join(self._weights_dir(), weight_files[-1])
