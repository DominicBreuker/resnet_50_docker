import sys
import os
import numpy as np
from model.resnet50 import build_resnet_50
from keras.preprocessing import image
from model.imagenet_utils import preprocess_input, decode_predictions

sys.path.append('/resnet_50/model')

IMAGE_PATH = '/resnet_50/model/test_images'

IMAGE_CLASS_LABELS = {
    '/resnet_50/model/test_images/cat1.jpg': "chow",
    '/resnet_50/model/test_images/cat2.jpg': "Egyptian_cat",
    '/resnet_50/model/test_images/cat3.jpg': "weasel",
    '/resnet_50/model/test_images/dog1.jpg': "vizsla",
    '/resnet_50/model/test_images/dog2.jpg': "German_shepherd",
    '/resnet_50/model/test_images/ipod.jpg': "iPod",
}


def predict(model, image_path, target_size=(224, 224)):
    image = load_image(image_path, target_size)
    return model.predict(image)


def load_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def load_test_image_paths():
    image_paths = []
    for filename in os.listdir(IMAGE_PATH):
        if filename.endswith(".jpg"):
            image_paths.append(os.path.join(IMAGE_PATH, filename))
    return sorted(image_paths)


def test_imagenet_classification():
    print("Testing classification mode with imagenet head...")

    model = build_resnet_50()
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


def test_imagenet_heatmap():
    print("Testing heatmap mode with imagenet head...")

    model = build_resnet_50(fully_convolutional=True)
    images = load_test_image_paths()
    for image_path in images:
        if "ipod" in image_path:
            dim = int(224 * 1)
            pred = predict(model, image_path, target_size=(dim, dim))

    heatmap = pred[0, :, :, 605]
    mean, std = np.mean(heatmap), np.std(heatmap)
    threshold = mean + 2*std
    assert(threshold > 0)
    heatmap[heatmap < threshold] = 0
    heatmap[heatmap >= threshold] = 1

    print("Binary ipod heatmap for ipod.jpg, scaled to {}x{}:"
          .format(dim, dim))
    print(heatmap)
    print("Compare ipod.jpg to see that it localizes the ipod!")

    true_heatmap = np.array([[0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  1.,  1.,  0.],
                             [0.,  0.,  0.,  0.,  1.,  1.,  0.],
                             [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    assert(np.allclose(true_heatmap, heatmap))


def test_custom_head_classification():
    print("Testing classfication mode with custom head...")

    model = build_resnet_50(cls_head='custom')
    images = load_test_image_paths()
    preds = []
    for image_path in images:
        preds.append(predict(model, image_path))
    predictios = np.vstack(preds)

    for i, pred in enumerate(predictios):
        current_image = images[i]
        is_cat = "cat" in current_image
        print("Image: {} | Cat score: {:3.5f} | No-cat score {:3.5f}"
              .format(current_image, pred[0], pred[1]))
        if is_cat:
            assert(pred[0] > 0.95)
            assert(pred[1] < 0.05)
        else:
            assert(pred[0] < 0.05)
            assert(pred[1] > 0.95)


def test_custom_head_heatmap():
    print("Testing fully convolutial mode with custom head...")

    model = build_resnet_50(fully_convolutional=True, cls_head='custom')
    images = load_test_image_paths()
    for image_path in images:
        if "cat3" in image_path:
            dim = int(224 * 1.3)
            pred = predict(model, image_path, target_size=(dim, dim))

    heatmap = pred[0, :, :, 0]
    mean, std = np.mean(heatmap), np.std(heatmap)
    threshold = mean + 2*std
    assert(threshold > 0)
    heatmap[heatmap < threshold] = 0
    heatmap[heatmap >= threshold] = 1
    print("Binary cat heatmap for cat3.jpg, scaled to {}x{}:".format(dim, dim))
    print(heatmap)
    print("Compare cat3.jpg to see that it localizes the cat!")

    true_heatmap = np.array([[0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    assert(np.allclose(true_heatmap, heatmap))


if __name__ == "__main__":
    print("")
    test_imagenet_classification()
    print("")
    test_imagenet_heatmap()
    print("")
    test_custom_head_classification()
    print("")
    test_custom_head_heatmap()
    print("")
