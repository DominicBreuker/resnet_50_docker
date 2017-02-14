[![Docker Automated build](https://img.shields.io/badge/docker%20build-automated-blue.svg)](https://hub.docker.com/r/dominicbreuker/resnet_50_docker/)

# ResNet 50 inside Docker
Easily extract image features from ResNet50 pre-trained on ImageNet.
Or just use it in prediction mode to get labels for input images.
Or even better, produce heatmaps to identify the location of objects in images.

# Quickstart

Clone this repo and run the `bin/extract_imagenet.sh` as well as `bin/train.sh`
scripts.
- `bin/extract_imagenet.sh` with extract ImageNet predictions for all images
  in this repository's `data` folder. The will appear in the `output` folder.
- `bin/train.sh` will train a custom clasification head for a cat vs no-cat
  classifier. It's weights will appear in `custom_heads/reference_model/weights`
  and for fitting, the code in `custom_heads/reference_model/methods/build_compile.py`
  will be used.



# How to use

With little to no customization and no installation, you can extract information
from images using a pre-trained ResNet model. Mount images into a Docker
container and specify what you want to generate. You can run this image in two
different modes: extraction and training.

Extraction allows you to apply the model to images to extract information.
By default, pre-trained ImageNet weights will be used to initialize ResNet.
You can use customly trained classification heads though (more on that later).
As output of the extraction, you can either get class labels of the final
classification layer or feature maps from the layer before the classification
layer. The latter allow to create nice heatmaps to locate objects.

Sometimes you want to apply ResNet with your own class labels. To do that,
you can train a custom classification head with your own labeled data.
Run this image in training mode to do so. This will first extract feature maps
from ResNet with ImageNet weights. These will be fed into a simple Keras model
acting as your custom classification head. This model will be saved for later
use in extractions.

## Extraction

The extraction script will look for images in the directory `/data` inside the
container. Mount your data folder accordingly. You can specify a file extension
of your images (e.g., `.jpg`). The script will search recursively for images
of that file type.

Outputs of the extraction script will be written to `/output` inside the
container. Thus, you should mount a host folder there as well to persist
outputs. Extraction generates two files. One is prefixed `index_resnet50`
and contains a numpy array of image names. The other is prefixed
`extractions_resnet50` and contains the main extraction output (i.e., class
labels or feature maps).

You can pass a number of options to the extraction script:
- -head (default = 'imagenet'): One of 'imagenet', 'custom' and 'none'.
  Defines how the final layer(s) of the model should be like. Use 'imagenet'
  for standard ImageNet classification. Use 'custom' for a customized model
  you creating with the training script. Or use 'none' to extract feature maps
  without a final layer.
- -fcn (default = False): Either True or False. If True, model will be converted
into a fully convolutional version. As a result, you can feed it images of any
size and it will apply the model to all 224x224 subframes of the image.
Suitable to create heatmaps of any size, either of class labels or feature maps.
- -e (default = 'jpg'): Can be any string. Defines the file extension of images.
- -hs (default = 224): Can be any integer. Defined the height to which all
images will be resized.
- -ws (default = 224): Can be any integer. Defined the width to which all
images will be resized.

Here is an example of how to start a container to extract a class label heatmap
from a customly trained classification head:

```bash
docker run --rm \
           -it \
           -v $DATA_DIR:/data \
           -v $OUTPUT_DIR:/output \
           dominicbreuker/resnet_50_docker:latest \
           /bin/sh -c "python /resnet_50/extracting.py -e jpg -w imagenet_refined -fcn True -head True -hs 300 -ws 300"

```

## Training custom classification heads

Like the extraction script, the training script will look for images in `/data`.
Inside this folder, create one subfolder per class with corresponding images
inside. The script first looks for extracted features in `/output`. If it does
not find them, it will create features using the extractor. Then it will train a
custom classification head and put it's weights into
`/resnet_50/model/custom_model/weights`. Make sure to mount a host directory
here to which the weights shall be saved.

Fitting is very basic and should be customized. You can do so by mounting your
own model compiling and fitting code to `/resnet_50/model/custom_model/methods`.
The default code should give you an idea how to customize. It looks like this:

```python
from keras.models import Model


class Compiler(object):
    def __init__(self, custom_head):
        self.custom_head = custom_head

    def compile(self, feature_input, output):
        model = Model(input=feature_input, output=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


class Fitter(object):
    def __init__(self, custom_head):
        self.custom_head = custom_head

    def fit(self, model, X, y):
        latest_weights_file = self.custom_head._latest_weights_file()
        if latest_weights_file is not None:
            self.custom_head.load_weights_into(model, False)
        model.fit(X, y)

```

Like for extracting, you also have to specify the file extension of images and
their height and width. For example, you can use the script as follows:

```bash
docker run --rm \
           -it \
           -v $DATA_DIR:/data \
           -v $OUTPUT_DIR:/output \
           -v $CUSTOM_MODEL_DIR:/resnet_50/model/custom_model/weights \
           -v $CUSTOM_COMPILE_FIT_DIR:/resnet_50/model/custom_model/methods \
           dominicbreuker/resnet_50_docker:latest \
           /bin/sh -c "python /resnet_50/training.py -e jpg"
```

To extract outputs of this custom classification head, just mount the `$CUSTOM_MODEL_DIR`
during extraction and set the flag `-head` to `custom`. The script will then
use the custom head and load the most recently created weights file. For
instance, run the following command to create heatmaps with your custom head:

```bash
docker run --rm \
           -it \
           -v $DATA_DIR:/data \
           -v $OUTPUT_DIR:/output \
           -v $CUSTOM_MODEL_DIR:/resnet_50/model/custom_model \
           dominicbreuker/resnet_50_docker:latest \
           /bin/sh -c "python /resnet_50/extracting.py -e jpg -head custom -fcn True -hs 300 -ws 300"
```

# Additional details

## Tests

To see if you are using the weights correctly, check out `/resnet_50/model/model_test.py`.
It will apply ResNet with pure ImageNet weights and a custom model to images in
`/resnet_50/model/test_images/*.jpg` to create predictions and heatmaps. This
script is run during the Docker image build to verify predictions are reasonable.

### Sources of test images:
- cat1.jpg: [Dwight Sipler](http://www.flickr.com/people/62528187@N00) from Stow, MA, USA, [Gillie hunting (2292639848)](https://commons.wikimedia.org/wiki/File:Gillie_hunting_(2292639848).jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/legalcode)
- cat2.jpg: The original uploader was [DrL](https://en.wikipedia.org/wiki/User:DrL) at [English Wikipedia](https://en.wikipedia.org/wiki/) [Blackcat-Lilith](https://commons.wikimedia.org/wiki/File:Blackcat-Lilith.jpg), [CC BY-SA 2.5
](https://creativecommons.org/licenses/by-sa/2.5/legalcode)
- cat3.jpg: [Pixabay picture](https://pixabay.com/en/cat-kitten-red-mackerel-tabby-1184743/) - Free for commercial use. No attribution required.
- dog1.jpg: HiSa Hiller, Schweiz, [Thai-Ridgeback](https://commons.wikimedia.org/wiki/File:Thai-Ridgeback.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode)
- dog2.jpg: [Military dog barking](https://commons.wikimedia.org/wiki/File:Military_dog_barking.JPG), in the [public domain](https://en.wikipedia.org/wiki/public_domain)
- ipod.jpg: [Marcus Quigmire](http://www.flickr.com/people/41896843@N00) from Florida, USA, [Baby Bloo taking a dip (3402460462)](https://commons.wikimedia.org/wiki/File:Baby_Bloo_taking_a_dip_(3402460462).jpg), [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/legalcode)
