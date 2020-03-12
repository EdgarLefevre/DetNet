import os
import skimage.io as io
import numpy as np


BASIC_H5_PATH = "/home/elefevre/Documents/Data_dypfish/basic.h5"


def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def create_dataset(path_img, path_label):
    print("Creating dataset...")
    dataset = []
    label_list = []
    file_list = list_files(path_img)
    for file in file_list:
        try:
            img = io.imread(path_img + file)
            label = io.imread(path_label + file)
            dataset.append((np.array(img) / np.amax(img)).reshape(-1, 512, 512, 1))
            label_list.append(label)
        except Exception as e:
            print(e)
            print("Image {} not found.".format(file))
    print("Created !")
    return np.array(dataset), np.array(label_list, dtype=np.bool)
