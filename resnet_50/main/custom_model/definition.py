from keras.models import Model
from keras.layers import Input, Flatten, Dense, AveragePooling2D


def build(input_shape, num_classes):
    feature_input = Input(shape=input_shape)
    x = AveragePooling2D((7, 7), name='avg_pool')(feature_input)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='custom_fc')(x)
    model = Model(input=feature_input, output=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def fit(model, X, y):
    model.fit(X, y)
