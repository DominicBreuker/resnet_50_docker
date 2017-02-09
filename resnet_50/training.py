import argparse
from main.extractor import Extractor
from main.trainer import Trainer
from model.resnet50 import build_resnet_50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extension", nargs="?", type=str,
                        default='jpg',
                        help="Look for images with this extension")
    parser.add_argument("-hs", "--height", nargs="?", type=int, default=224,
                        help="Image height")
    parser.add_argument("-ws", "--width", nargs="?", type=int, default=224,
                        help="Image width")
    return parser.parse_args()


def extract_features(extension, img_size):
    extractor = Extractor(extension)
    image_files, features = extractor.load_latest_extractions()
    if image_files is None or features is None:
        name = "{}x{}_{}".format(img_size[0], img_size[1], "feature-maps")
        model = build_resnet_50(include_top=False)
        image_files, features = extractor.extract(name, model, img_size)
    print(image_files)
    print(features.shape)
    return image_files, features


def train_model(image_files, features):
    trainer = Trainer(image_files, features)
    model = trainer.train()
    print(model)
    return model


if __name__ == "__main__":
    args = parse_args()
    img_size = (args.height, args.width)

    image_files, features = extract_features(args.extension, img_size)
    model = train_model(image_files, features)
