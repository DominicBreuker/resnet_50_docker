import os
import numpy as np
from sets import Set
from model.custom_model.definition import CustomHead


class Trainer(object):
    def __init__(self, image_files, features):
        self.image_files = image_files
        self.features = features
        self.classes = self._get_classes()
        self.input_shape = self._get_input_shape()
        self.custom_head = CustomHead(len(self.classes))

    def train(self):
        class_labels = self._one_hot_class_labels()
        model = self._build_model()
        self._fit_model(model, class_labels)
        self._save_model(model)
        return model

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
        return self.custom_head.build(self.input_shape)

    def _fit_model(self, model, class_labels):
        self.custom_head.fit(model, self.features, class_labels)

    def _save_model(self, model):
        self.custom_head.save(model)
