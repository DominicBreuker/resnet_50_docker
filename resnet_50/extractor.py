import os
import argparse
import numpy as np
from tqdm import tqdm
from model.resnet50 import ResNet50
from model.imagenet_utils import preprocess_input, decode_predictions
from result_saver.npy import save_results
from keras.preprocessing import image

IMAGE_DIRECTORY = '/data'
OUTPUT_DIRECTORY = '/output'


def get_image_files(extension):
    files = []
    print("Looking for '.{}' images in: {}".format(extension, IMAGE_DIRECTORY))
    for root, directories, filenames in os.walk(IMAGE_DIRECTORY):
        for filename in filenames:
            if filename.endswith(".{}".format(extension)):
                files.append(os.path.join(root, filename))
    print("{} images found...".format(len(files)))
    return files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extension", nargs="?", type=str,
                        default='jpg',
                        help="Look for images with this extension")
    parser.add_argument("-m", "--mode", nargs="?",
                        choices=['predict', 'feature'],
                        default='predict',
                        help="Run in on of these modes. 'predict' for predicting \
                        textual class label, 'feature' for 1x2048 feature \
                        vecrors.")
    parser.add_argument("-hs", "--height", nargs="?", type=int, default=224,
                        help="Image height - can be changed only in \
                        'feature' mode")
    parser.add_argument("-ws", "--width", nargs="?", type=int, default=224,
                        help="Image width - can be changed only in \
                        'feature' mode")
    return parser.parse_args()


def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def extract_class_labels(model, image_file, image_size):
    image = load_image(image_file)
    predictions = model.predict(image)
    return [p[1] for p in decode_predictions(predictions)[0]]


def extract_features(model, image_file, image_size):
    image = load_image(image_file)
    return model.predict(image)[0]


def strip_data_folder_from_image_paths(image_files):
    return ["/".join(image_file.split("/")[2:]) for image_file in image_files]


if __name__ == "__main__":
    args = parse_args()

    if (args.mode != 'feature') and \
       ((args.height != 224) or (args.width != 224)):
        raise Exception("Custom size can only be used in feature mode")

    input_size = (args.height, args.width)
    image_files = get_image_files(args.extension)
    if args.mode == 'predict':
        model = ResNet50(include_top=True, weights='imagenet')
        extractor = extract_class_labels
    elif args.mode == 'feature':
        model = ResNet50(include_top=False, weights='imagenet')
        extractor = extract_features
    else:
        raise Exception("unsupported mode: {}".format(args.mode))

    extractions = []
    for image_file in tqdm(image_files):
        extractions.append(extractor(model, image_file, input_size))
    extractions = np.vstack(extractions)

    image_files = strip_data_folder_from_image_paths(image_files)
    save_results(args, image_files, extractions, OUTPUT_DIRECTORY)
