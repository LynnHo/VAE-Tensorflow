from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import multiprocessing
import os
import struct
import subprocess

import numpy as np
import tensorflow as tf
from tflib.data.memory_data import MemoryData


_N_CPU = multiprocessing.cpu_count()


def unzip_gz(file_name):
    unzip_name = file_name.replace('.gz', '')
    gz_file = gzip.GzipFile(file_name)
    open(unzip_name, 'w+').write(gz_file.read())
    gz_file.close()


def mnist_download(download_dir):
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = url_base + file_name
        save_path = os.path.join(download_dir, file_name)
        cmd = ['curl', url, '-o', save_path]
        print('Downloading ', file_name)
        if not os.path.exists(save_path):
            subprocess.call(cmd)
        else:
            print('%s exists, skip!' % file_name)


def mnist_load(data_dir, split='train'):
    """Load MNIST dataset, modified from https://gist.github.com/akesling/5358964.

    Returns:
        `imgs`, `lbls`, `num`.

        `imgs` : [-1.0, 1.0] float64 images of shape (N * H * W).
        `lbls` : Int labels of shape (N,).
        `num`  : # of datas.
    """
    mnist_download(data_dir)

    if split == 'train':
        fname_img = os.path.join(data_dir, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    elif split == 'test':
        fname_img = os.path.join(data_dir, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("split must be 'test' or 'train'")

    if not os.path.exists(fname_img):
        unzip_gz(fname_img + '.gz')
    if not os.path.exists(fname_lbl):
        unzip_gz(fname_lbl + '.gz')

    with open(fname_lbl, 'rb') as flbl:
        struct.unpack('>II', flbl.read(8))
        lbls = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, _, rows, cols = struct.unpack('>IIII', fimg.read(16))
        imgs = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbls), rows, cols)
        imgs = imgs / 127.5 - 1

    return imgs, lbls, len(lbls)


class Mnist(MemoryData):

    def __init__(self, data_dir, batch_size, split='train', prefetch_batch=_N_CPU + 1, drop_remainder=True, filter=None,
                 map_func=None, num_threads=_N_CPU, shuffle=True, buffer_size=4096, repeat=-1, sess=None):
        imgs, lbls, _ = mnist_load(data_dir, split)
        imgs.shape = imgs.shape + (1,)

        imgs_pl = tf.placeholder(tf.float32, imgs.shape)
        lbls_pl = tf.placeholder(tf.int64, lbls.shape)

        memory_data_dict = {'img': imgs_pl, 'lbl': lbls_pl}

        self.feed_dict = {imgs_pl: imgs, lbls_pl: lbls}
        super(Mnist, self).__init__(memory_data_dict, batch_size, prefetch_batch, drop_remainder, filter,
                                    map_func, num_threads, shuffle, buffer_size, repeat, sess)

    def reset(self):
        super(Mnist, self).reset(self.feed_dict)

if __name__ == '__main__':
    import imlib as im
    from tflib import session
    sess = session()
    mnist = Mnist('/tmp', 5000, repeat=1, sess=sess)
    print(len(mnist))
    for batch in mnist:
        print(batch['lbl'][-1])
        im.imshow(batch['img'][-1].squeeze())
        im.show()
    sess.close()
