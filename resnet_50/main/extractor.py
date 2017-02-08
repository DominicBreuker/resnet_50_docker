import os
import numpy as np
from keras.preprocessing import image
from model.imagenet_utils import preprocess_input
from tqdm import tqdm
from result_saver.npy import save_results, load_latest_results
from constants import IMAGE_DIRECTORY, OUTPUT_DIRECTORY


class Extractor(object):
    def __init__(self, extension):
        self.extension = extension

    def load_latest_extractions(self):
        return load_latest_results(OUTPUT_DIRECTORY)

    def extract(self, name, model, image_size):
        image_files = self._image_files()
        extractions = []
        for image_file in tqdm(image_files):
            image = self._load_image(image_file, image_size)
            extractions.append(model.predict(image))
        extractions = np.vstack(extractions)

        save_results(name, image_files, extractions, OUTPUT_DIRECTORY)
        return image_files, extractions

    def _image_files(self):
        files = []
        print("Looking for '.{}' images in: {}"
              .format(self.extension, IMAGE_DIRECTORY))
        for root, directories, filenames in os.walk(IMAGE_DIRECTORY):
            for filename in filenames:
                if filename.endswith(".{}".format(self.extension)):
                    files.append(os.path.join(root, filename))
        print("{} images found...".format(len(files)))
        return [image_file[len(IMAGE_DIRECTORY)+1:] for image_file in files]

    def _load_image(self, image_path, image_size):
        img = image.load_img(os.path.join(IMAGE_DIRECTORY, image_path),
                             target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)
