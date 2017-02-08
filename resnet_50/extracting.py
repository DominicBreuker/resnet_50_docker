import os
import argparse
from main.extractor import Extractor
from main.trainer import Trainer
from model.resnet50 import build_resnet_50
from main.constants import OUTPUT_DIRECTORY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", nargs="?",
                        choices=['imagenet', 'imagenet_refined'],
                        default='imagenet',
                        help="Choose the weights to use for extraction. \
                              Use 'imagenet' for pretrained imagenet weights \
                              Use 'imagenet_refined' for imagenet weights \
                              with a customly trained classification head.")
    parser.add_argument("-fcn", "--fully_convolutional", nargs="?", type=bool,
                        default=False,
                        help="Transform to FCN. If True, you can feed it \
                              images of any size. You will get a heatmap \
                              output rather than a classification for the \
                              entire image.")
    parser.add_argument("-head", "--include_head", nargs="?", type=bool,
                        default=True,
                        help="Include final classification layer. If False \
                              You will get the feature map output of the \
                              layer right before the final classification. \
                              Useful if you want to extract features to \
                              train your own model on.")
    parser.add_argument("-e", "--extension", nargs="?", type=str,
                        default='jpg',
                        help="Look for images with this extension")
    parser.add_argument("-hs", "--height", nargs="?", type=int, default=224,
                        help="Image height")
    parser.add_argument("-ws", "--width", nargs="?", type=int, default=224,
                        help="Image width")
    args = parser.parse_args()

    if args.weights == 'imagenet' \
       and img_size != (224, 224):
        raise Exception("ERROR: Imagenet requires images of size 224x224!")

    if not args.include_head and args.weights == 'imagenet_refined':
        raise Exception("ERROR: Refined imagenet mode implies a \
                         classfication head, but you specified to \
                         not use one!")

    return args


def get_latest_custom_head(img_size):
    filename_base = Trainer.model_filename_base(img_size)
    model_files = []
    for root, directories, filenames in os.walk(OUTPUT_DIRECTORY):
        for filename in filenames:
            if filename.endswith(".h5"):
                if filename.startswith(filename_base):
                    model_files.append(os.path.join(OUTPUT_DIRECTORY,
                                                    filename))
    model_files.sort()
    return model_files[-1]


def build_model(include_top, fully_convolutional, custom_head, img_size):
    if custom_head:
        top_weights = 'imagenet'
    else:
        top_weights = get_latest_custom_head(img_size)
        if top_weights is None:
            raise Exception("ERROR: no custom head model for image size {}x{}"
                            .format(img_size[0], img_size[1]))
    return build_resnet_50(include_top=include_top,
                           fully_convolutional=fully_convolutional,
                           top_weights=top_weights)


def extract(name, model, extension, img_size):
    extractor = Extractor(extension)
    image_files, features = extractor.extract(name, model, img_size)


def filename_base(args):
    return "{}x{}-FCN_{}-{}-HEAD_{}" \
           .format(args.height, args.width, args.fully_convolutional,
                   args.weights, args.include_head)


if __name__ == "__main__":
    args = parse_args()
    img_size = (args.height, args.width)

    model = build_model(args.include_head, args.fully_convolutional,
                        args.weights == 'imagenet_refined', img_size)
    name = filename_base(args)
    extract(name, model, args.extension, img_size)
