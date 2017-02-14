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
