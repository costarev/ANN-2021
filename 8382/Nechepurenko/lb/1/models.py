from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
import pandas as pd


class ModelBuilder:

    def __init__(self):
        self._compile_config = {
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
        }

        self._fit_config = {
            "epochs": 75,
            "batch_size": 10,
            "validation_split": 0.1
        }

        self._layers = [Dense(4, activation='relu'), Dense(3, activation='softmax')]

    def _decorate(self, fn):
        def fit(x, y):
            return fn(x, y, **self._fit_config, verbose=False)

        return fit

    def set_layers(self, layers):
        self._layers = layers
        return self

    def set_params(self, **params):
        for param_name, value in params.items():
            assert param_name in [*self._compile_config, *self._fit_config], "Unknown parameter"
            if param_name in self._compile_config:
                self._compile_config[param_name] = value
            else:
                self._fit_config[param_name] = value
        return self

    def build(self):
        model = Sequential()
        for layer in self._layers:
            model.add(layer)
        model.compile(**self._compile_config)
        model.fit = self._decorate(model.fit)
        return model


def get_models_to_compare():

    def cast_to_numeric_type(pair):

        def try_cast(type_t, value_to_cast):
            try:
                value_to_cast = type_t(value_to_cast)
            except ValueError:
                return value_to_cast, False
            else:
                return value_to_cast, True

        param, value = pair
        for type_ in [int, float]:
            value, ok = try_cast(type_, value)
            if ok:
                return param, value
        return param, value

    def parse_config(config_str):
        return dict(map(cast_to_numeric_type, map(lambda x: x.split(':'), config_str.split(';'))))

    def parse_layers(layers_str):
        return [Dense(int(count), activation=act_fn) for count, act_fn in
                map(lambda x: x.split(':'), layers_str.split(';'))]

    models = []
    for config_str, layers_str in pd.read_csv("models.csv", header=None).values:
        models.append((parse_config(config_str), parse_layers(layers_str)))
    return models
