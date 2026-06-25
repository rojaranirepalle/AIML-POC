import numpy as np


def predict(model, values):

    arr = np.array(values).reshape(1, -1)

    return model.predict(arr)[0]