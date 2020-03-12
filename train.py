#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorflow.keras as keras
import utils.data as data
import detnet
import tensorflow as tf
import numpy as np


def train(path_images, path_labels):
    file_path = "saved_models/detnet.h5"
    dataset, label = data.create_dataset(path_images, path_labels)
    print(np.shape(dataset))
    checkpoint = keras.callbacks.ModelCheckpoint(file_path, save_best_only=True, mode="min")
    earlystopper = keras.callbacks.EarlyStopping(patience=opt.patience, verbose=1, min_delta=0.00001,
                                                 restore_best_weights=True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = detnet.detnet((512, 512, 1))
        model_dice = detnet.soft_dice_loss
        model.compile(loss=model_dice, optimizer=tf.keras.optimizers.Adam(lr=0.001, amsgrad=True))
        model.fit(dataset, label,
                  validation_split=0.2,
                  batch_size=opt.batch_size,
                  epochs=opt.n_epochs,
                  callbacks=[earlystopper, checkpoint])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--size", type=int, default=512, help="Size of the image, one number")
    parser.add_argument("--patience", type=int, default=10, help="Set patience value for early stopper")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    opt = get_args()
    # train_detect(opt, opt.model_name)
    # train()
    train(path_images="../Data/data_dypfish/fish/spot_mask/train/", path_labels="../Data/data_dypfish/fish/spot_mask/mask/train/")
    # train(path_images="/home/elefevre/Documents/Data_dypfish/data/fish/spot/train/",
    #       path_labels="/home/elefevre/Documents/Data_dypfish/data/fish/spot/mask/train/")