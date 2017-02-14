import numpy as np
import h5py

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model

from model.custom_model.definition import CustomHead


def build_resnet_50(fully_convolutional=False, main_weights='imagenet',
                    cls_head='imagenet'):
    '''Builds a ResNet50 model with Keras.
    Arguments:
    - fully_convolutional: If True, the top dense layer will be
                           convolutionized.
    - main_weights: Set to 'imagenet' to load image net weights or to None
                    to load no weights.
    - cls_head:  Set to 'imagenet' to load image net head with weights,
                 or pass in 'custom' to to create a customly defined head,
                 or set to None if you do not want a classfication head.
    '''
    # load main part
    if cls_head is None or fully_convolutional:
        img_input, x = build_resnet_50_main((None, None, 3))
    else:
        img_input, x = build_resnet_50_main((224, 224, 3))

    # load head part
    if cls_head == 'imagenet':
        x = add_top_layer_imagenet(x, fully_convolutional)
    elif cls_head == 'custom':
        custom_head = CustomHead()
        x = add_top_layer_custom_head(x, fully_convolutional, custom_head)

    model = Model(img_input, x)

    # load main weights
    if main_weights is not None:
        load_main_weights(model, main_weights)

    # load head weights
    if cls_head == 'imagenet':
        load_top_layer_weights_imagenet(model, fully_convolutional)
    elif cls_head == 'custom':
        load_top_layer_weights_custom(model, fully_convolutional, custom_head)

    return model


def add_top_layer_imagenet(x, fully_convolutional):
    if fully_convolutional:
        layer_name = "fc{}_conv".format(1000)
        x = Convolution2D(1000, 1, 1, border_mode='same', name=layer_name)(x)
    else:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        layer_name = "fc{}".format(1000)
        x = Dense(1000, activation='softmax', name=layer_name)(x)
    return x


def add_top_layer_custom_head(x, fully_convolutional, custom_head):
    return custom_head.add_layers(x, fully_convolutional)


def load_main_weights(model, main_weights):
    if main_weights == 'imagenet':
        weights_path = '/root/.keras/models/resnet_50_tf.h5'
        model.load_weights(weights_path, by_name=True)
    else:
        model.load_weights(main_weights, by_name=True)


def load_top_layer_weights_imagenet(model, fully_convolutional):
    W, b = get_top_layer_weights_imagenet()
    if fully_convolutional:
        W = convolutionize(W)
        layer_name = 'fc1000_conv'
    else:
        layer_name = 'fc1000'
    model.get_layer(layer_name).set_weights([W, b])


def load_top_layer_weights_custom(model, fully_convolutional, custom_head):
    custom_head.load_weights_into(model, fully_convolutional)


def convolutionize(W):
    return np.expand_dims(np.expand_dims(W, axis=0), axis=0)


def get_top_layer_weights_imagenet():
    f = get_imagenet_weighs()
    W, b = f['fc1000']['fc1000_W:0'].value, f['fc1000']['fc1000_b:0'].value
    return W, b


def get_imagenet_weighs():
    weights_path = '/root/.keras/models/resnet_50_tf.h5'
    return get_weighs(weights_path)


def get_weighs(weights_path):
    return h5py.File(weights_path, 'r')


def get_top_layer_weights_custom(custom_weights):
    f = get_weighs(custom_weights)
    W = f['model_weights']['custom_fc']['custom_fc_W:0'].value
    b = f['model_weights']['custom_fc']['custom_fc_b:0'].value
    return W, b


def build_resnet_50_main(input_shape):
    img_input = Input(shape=input_shape)

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    return img_input, x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters,
               stage, block, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis,
                                  name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x
