from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback

import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl
import utils


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', dest='experiment_name', help='experiment_name')
args_ = parser.parse_args()
with open('./output/%s/setting.txt' % args_.experiment_name) as f:
    args = json.load(f)

z_dim = args["z_dim"]

dataset_name = args["dataset_name"]
model_name = args["model_name"]
experiment_name = args_.experiment_name

# dataset and models
_, img_shape, _ = utils.get_dataset(dataset_name)
_, Dec = utils.get_models(model_name)
Dec = partial(Dec, channels=img_shape[2])


# ==============================================================================
# =                                    graph                                   =
# ==============================================================================

# input
z_sample = tf.placeholder(tf.float32, [None, z_dim])

# sample
img_sample = Dec(z_sample, is_training=False)


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# session
sess = tl.session()

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    raise Exception(' [*] No checkpoint!')

# train
try:
    z_ipt_sample_ = np.random.normal(size=[10, z_dim])
    for i in range(z_dim):
        z_ipt_sample = np.copy(z_ipt_sample_)
        img_opt_samples = []
        for v in np.linspace(-3, 3, 10):
            z_ipt_sample[:, i] = v
            img_opt_samples.append(sess.run(img_sample, feed_dict={z_sample: z_ipt_sample}).squeeze())

        save_dir = './output/%s/sample_traversal' % experiment_name
        pylib.mkdir(save_dir)
        im.imwrite(im.immerge(np.concatenate(img_opt_samples, axis=2), 10), '%s/traversal_d%d.jpg' % (save_dir, i))
except:
    traceback.print_exc()
finally:
    sess.close()
