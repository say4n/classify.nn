import scipy.io
import glob
import numpy as np

def get_data(path, *, debug=False):
    data = {
        'A': [],
        'AA': [],
        'Ae': [],
        'AeA': [],
        'EE': [],
        'OO': [],
        'UU': []
    }

    indices = np.array([1, 2, 7, 8, 9, 10, 19]) - 1

    if debug:
        breakpoint()

    for mat in glob.glob(path):
        tmp = scipy.io.loadmat(mat)
        classname = mat.split('/')[1].split('_')[1].split('.')[0]

        avoiders = ['__header__', '__version__', '__globals__']
        key = [k for k in tmp.keys() if k not in avoiders][0]

        data[classname].append(tmp[key])
        if debug:
            print(f"{classname} :: {key}")

    X_train = []
    y_train = []

    for category in data:
        if category == 'A':
            label = [1, 0, 0, 0, 0, 0, 0]
        elif category == 'AA':
            label = [0, 1, 0, 0, 0, 0, 0]
        elif category == 'Ae':
            label = [0, 0, 1, 0, 0, 0, 0]
        elif category == 'AeA':
            label = [0, 0, 0, 1, 0, 0, 0]
        elif category == 'EE':
            label = [0, 0, 0, 0, 1, 0, 0]
        elif category == 'OO':
            label = [0, 0, 0, 0, 0, 1, 0]
        elif category == 'UU':
            label = [0, 0, 0, 0, 0, 0, 1]

        label = np.array(label)

        for sample in data[category]:
            sample = sample[:, indices]
            # breakpoint()
            X_train.append(sample.reshape(2500 * len(indices),))
            y_train.append(label)

    return np.array(X_train), np.array(y_train)