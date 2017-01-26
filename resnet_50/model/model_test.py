import os
import numpy as np
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

IMAGE_PATH = '/resnet_50/model/test_images'

IMAGE_CLASS_LABELS = {
    '/resnet_50/model/test_images/cat1.jpg': "chow",
    '/resnet_50/model/test_images/cat2.jpg': "Egyptian_cat",
    '/resnet_50/model/test_images/dog1.jpg': "vizsla",
    '/resnet_50/model/test_images/dog2.jpg': "German_shepherd",
    '/resnet_50/model/test_images/ipod.jpg': "iPod",
}


def test():
    model = load_model()
    images = load_test_image_paths()
    preds = []
    for image_path in images:
        preds.append(predict(model, image_path))
    predictios = np.vstack(preds)

    for i, top5 in enumerate(decode_predictions(predictios)):
        current_image = images[i]
        current_class_label = IMAGE_CLASS_LABELS[current_image]
        class_labels_predictions = [pred[1] for pred in top5]
        print("Image: {} | Actual: {} | Top5 predictions: {}"
              .format(current_image, current_class_label,
                      class_labels_predictions))
        assert(current_class_label in class_labels_predictions)


def load_model():
    return ResNet50(weights='imagenet')


def predict(model, image_path):
    image = load_image(image_path)
    return model.predict(image)


def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def load_test_image_paths():
    image_paths = []
    for filename in os.listdir(IMAGE_PATH):
        if filename.endswith(".jpg"):
            image_paths.append(os.path.join(IMAGE_PATH, filename))
    return sorted(image_paths)


if __name__ == "__main__":
    test()
