import os
import argparse
from sets import Set
from datetime import datetime
import numpy as np
from tqdm import tqdm
from model.resnet50 import ResNet50
from model.imagenet_utils import preprocess_input, decode_predictions
from result_saver.npy import save_results, load_latest_results
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model, load_model

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


def extract_from_network(args):
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
    return image_files, extractions


def train_custom_top(image_files, extractions):
    print(image_files)
    classes = get_classes(image_files)
    print(classes)
    class_labels = get_one_hot_class_labels(classes, image_files)
    print(class_labels)
    print("---")
    print(extractions.shape)
    input_shape = extractions.shape[1:]
    model = build_model(len(classes), input_shape)
    fit_model(model, extractions, class_labels)
    save_model(model, input_shape, OUTPUT_DIRECTORY)
    return model


def get_classes(image_files):
    ''' Infers classes from the image paths.
    Assumes images are put into folders like this:
    - class_1/image1.jpp
    - class_1/image2.jpp
    - class_2/image1.jpp
    - class_2/image2.jpp
    '''
    classes = Set()
    for image_file in image_files:
        classes.add(get_class(image_file))
    return np.array(sorted(classes))


def get_class(image_file):
    return image_file.split(os.sep)[0]


def get_one_hot_class_labels(classes, image_files):
    class_labels = np.zeros((len(image_files), len(classes)))
    for i, image_file in enumerate(image_files):
        image_class = get_class(image_file)
        class_labels[i, np.where(classes == image_class)[0][0]] = 1
    return class_labels


def build_model(num_classes, input_shape):
    feature_input = Input(shape=input_shape)
    x = Flatten()(feature_input)
    x = Dense(num_classes, activation='softmax', name='custom_fc')(x)
    model = Model(input=feature_input, output=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def fit_model(model, data, labels):
    model.fit(data, labels)


def model_file(input_shape, out_dir):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return os.path.join(out_dir,
                        "custom_top_{}x{}_{}.h5".format(input_shape[0],
                                                        input_shape[1],
                                                        current_time))


def save_model(model, input_shape, out_dir):
    filename = model_file(input_shape, out_dir)
    print("Saving custom top model to {}".format(filename))
    model.save(filename)


def load_latest_model(extractions, out_dir):
    input_shape = extractions.shape[1:]
    model_files = []
    for root, directories, filenames in os.walk(out_dir):
        for filename in filenames:
            if filename.endswith(".h5"):
                if filename.startswith('custom_top_{}x{}'.format(input_shape[0],
                                                                 input_shape[1])):
                    model_files.append(os.path.join(out_dir, filename))
    model_files.sort()
    if len(model_files) == 0:
        return None
    else:
        print("\nLoading existing custom top model:\n{}\n"
              .format(model_files[-1]))
        return load_model(os.path.join(out_dir, model_files[-1]))


def predict(model, extractions, out_dir):
    input_shape = extractions.shape[1:]
    predictions = model.predict(extractions)
    print(predictions)
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    predictions_filename = "predictions_resnet50_custom_head_{}x{}_{}"\
        .format(input_shape[0], input_shape[1], current_time)
    np.save(os.path.join(out_dir, predictions_filename), predictions)
    print("Predictions saved ... load with numpy: 'np.load('{}.npy')'"
          .format(predictions_filename))


if __name__ == "__main__":
    args = parse_args()

    image_files, extractions = load_latest_results(OUTPUT_DIRECTORY)
    if image_files is None or extractions is None:
        image_files, extractions = extract_from_network(args)

    model = load_latest_model(extractions, OUTPUT_DIRECTORY)
    if model is None:
        model = train_custom_top(image_files, extractions)

    predict(model, extractions, OUTPUT_DIRECTORY)
