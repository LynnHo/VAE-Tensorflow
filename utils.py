from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import glob as glob

import models
import pylib
import tensorflow as tf
import tflib as tl


def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        # dataset
        pylib.mkdir('./data/mnist')
        Dataset = partial(tl.Mnist, data_dir='./data/mnist', repeat=1)

        # shape
        img_shape = [28, 28, 1]

        # index func
        def get_imgs(batch):
            return batch['img']

        return Dataset, img_shape, get_imgs

    elif dataset_name == 'celeba':
        # dataset
        def _map_func(img):
            crop_size = 108
            re_size = 64
            img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
            img = tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            return img

        paths = glob.glob('./data/celeba/img_align_celeba/*.jpg')
        Dataset = partial(tl.DiskImageData, img_paths=paths, repeat=1, map_func=_map_func)

        # shape
        img_shape = [64, 64, 3]

        # index func
        def get_imgs(batch):
            return batch

        return Dataset, img_shape, get_imgs


def get_models(model_name):
    return getattr(models, model_name)()
