import tensorflow as tf
import tensorflow.keras as keras
import detnet
import numpy as np
import utils.data as data
import skimage.io as io


def predict_seg(dataset, path_model_seg):
    model_seg = keras.models.load_model(path_model_seg, custom_objects={'dice loss': detnet.soft_dice_loss})
    return (model_seg.predict(dataset) > 0.5).astype(np.uint8).reshape(len(dataset), 512, 512, 1)


if __name__ == "__main___":
    dataset, labels = data.create_dataset("../Data/data_dypfish/fish/spot_mask/test/",
                                          "../Data/data_dypfish/fish/spot_mask/mask/test/")
    print(dataset[0])
    print(np.shape(dataset[0]))
    mask = predict_seg(dataset[0], "saved_models/detnet.h5")
    print(mask)
    io.imsave("mask_pred.png", mask)
