import sys
import argparse
from main.extractor import Extractor
from model.resnet50 import build_resnet_50

sys.path.append('/resnet_50/model')
sys.path.append('/resnet_50/main')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-head", "--cls_head", nargs="?",
                        choices=['imagenet', 'custom', 'none'],
                        default='imagenet',
                        help="Choose the head ...")
    parser.add_argument("-fcn", "--fully_convolutional", nargs="?", type=bool,
                        default=False,
                        help="Transform to FCN. If True, you can feed it \
                              images of any size. You will get a heatmap \
                              output rather than a classification for the \
                              entire image.")
    parser.add_argument("-e", "--extension", nargs="?", type=str,
                        default='jpg',
                        help="Look for images with this extension")
    parser.add_argument("-hs", "--height", nargs="?", type=int, default=224,
                        help="Image height")
    parser.add_argument("-ws", "--width", nargs="?", type=int, default=224,
                        help="Image width")
    args = parser.parse_args()

    return args


def extract(name, model, extension, img_size):
    extractor = Extractor(extension)
    image_files, features = extractor.extract(name, model, img_size)


def filename_base(args):
    return "{}x{}-FCN_{}-{}" \
           .format(args.height, args.width, args.fully_convolutional,
                   args.cls_head)


if __name__ == "__main__":
    args = parse_args()
    img_size = (args.height, args.width)

    model = build_resnet_50(args.fully_convolutional, 'imagenet',
                            args.cls_head)
    name = filename_base(args)
    extract(name, model, args.extension, img_size)
