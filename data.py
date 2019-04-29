import h5py
import matplotlib.pyplot as plt

import numpy as np


def load_from_H5(file_path):
    h5_file = h5py.File(file_path)
    data = h5_file.get('data')
    label = h5_file.get('label')
    return np.array(data), np.array(label).squeeze()
