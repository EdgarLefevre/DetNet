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


def train(path_images, path_labels):
    file_path = "saved_models/detnet.h5"
    dataset, label = data.create_dataset(path_images, path_labels)

    checkpoint = keras.callbacks.ModelCheckpoint(file_path, save_best_only=True)
    earlystopper = keras.callbacks.EarlyStopping(patience=opt.patience, verbose=1, min_delta=0.00001,
                                                 restore_best_weights=True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = detnet.detnet((512, 512, 1))
        model_dice = detnet.dice_loss(smooth=1e-5, thresh=0.5)
        model.compile(loss=model_dice, optimizer=tf.keras.optimizers.Adam(lr=0.001, amsgrad=True))
        model.fit(dataset, label,
                  validation_split=0.2,
                  batch_size=opt.batch_size,
                  epochs=opt.n_epochs,
                  callbacks=[earlystopper, checkpoint])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=300, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--model_name", type=str, default="small++",
                        help="Name of the model you want to train (detection, small++)")
    parser.add_argument("--meta", type=bool, default=True, help="True if we want to segment metastasis")
    parser.add_argument("--weighted", type=bool, default=False, help="Use weighted model (default False)")
    parser.add_argument("--size", type=int, default=128, help="Size of the image, one number")
    parser.add_argument("--w1", type=int, default=2, help="weight inside")
    parser.add_argument("--w2", type=int, default=4, help="Weight border")
    parser.add_argument("--patience", type=int, default=10, help="Set patience value for early stopper")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    opt = get_args()
    # train_detect(opt, opt.model_name)
    # train()
    train(path_images=gv.meta_path_img, path_labels=gv.meta_path_lab)
