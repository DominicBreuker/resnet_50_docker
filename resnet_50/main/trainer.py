import os
import sys
import numpy as np
from sets import Set
from datetime import datetime
from constants import OUTPUT_DIRECTORY
from .custom_model.definition import build, fit
from keras.models import load_model

sys.path.append('/resnet_50/main/custom_model')


class Trainer(object):
    def __init__(self, image_files, features):
        self.image_files = image_files
        self.features = features
        self.classes = self._get_classes()
        self.input_shape = self._get_input_shape()

    def train(self):
        class_labels = self._one_hot_class_labels()
        model = self._build_model()
        self._fit_model(model, class_labels)
        self._save_model(model)
        return model

    def load_latest_model(self):
        model_files = []
        for root, directories, filenames in os.walk(OUTPUT_DIRECTORY):
            for filename in filenames:
                if filename.endswith(".h5"):
                    if filename.startswith(self._model_filename_base()):
                        model_files.append(os.path.join(OUTPUT_DIRECTORY,
                                                        filename))
        model_files.sort()
        if len(model_files) == 0:
            return None
        else:
            print("\nLoading existing custom top model:\n{}\n"
                  .format(model_files[-1]))
            return load_model(os.path.join(OUTPUT_DIRECTORY, model_files[-1]))

    def _get_classes(self):
        ''' Infers classes from the image paths.
        Assumes images are put into folders like this:
        - class_1/image1.jpp
        - class_1/image2.jpp
        - class_2/image1.jpp
        - class_2/image2.jpp
        '''
        classes = Set()
        for image_file in self.image_files:
            classes.add(self._get_class(image_file))
        return np.array(sorted(classes))

    def _get_class(self, image_file):
        return image_file.split(os.sep)[0]

    def _get_input_shape(self):
        return self.features.shape[1:]

    def _one_hot_class_labels(self):
        class_labels = np.zeros((len(self.image_files), len(self.classes)))
        for i, image_file in enumerate(self.image_files):
            image_class = self._get_class(image_file)
            class_labels[i, np.where(self.classes == image_class)[0][0]] = 1
        return class_labels

    def _build_model(self):
        return build(self.input_shape, len(self.classes))

    def _fit_model(self, model, class_labels):
        fit(model, self.features, class_labels)

    def _model_path(self):
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = "{}_{}.h5".format(self._model_filename_base(), current_time)
        return os.path.join(OUTPUT_DIRECTORY, filename)

    @staticmethod
    def model_filename_base(input_shape):
        return "custom_top_{}x{}".format(input_shape[0], input_shape[1])

    def _model_filename_base(self):
        return self.model_filename_base(self.input_shape)

    def _save_model(self, model):
        print("Saving custom top model to {}".format(self._model_path()))
        model.save(self._model_path())
