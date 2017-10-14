from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import utils
import traceback
import numpy as np
import tensorflow as tf
import models_64x64 as models

from functools import partial


""" param """
epoch = 100
batch_size = 64
lr = 0.0002
z_dim = 2048
gpu_id = 2

''' data '''
# you should prepare your own data in ./data/img_align_celeba
# celeba original size is [218, 178, 3]


def preprocess_fn(img):
    crop_size = 108
    re_size = 64
    img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
    img = tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)

    img = tf.subtract(img, tf.reduce_min(img)) / tf.subtract(tf.reduce_max(img), tf.reduce_min(img)) * 2 - 1  # -> [-1.0, 1.0]
    return img

img_paths = glob.glob('./data/img_align_celeba/*.jpg')
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)


""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' models '''
    encoder = partial(models.encoder, z_dim=z_dim)
    decoder = models.decoder

    def enc_dec(img, reuse=True, training=False):
        # encode
        z_mu, z_log_sigma_sq = encoder(img, reuse=reuse, training=training)

        # sample
        epsilon = tf.random_normal(tf.shape(z_mu))
        if training:
            z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * epsilon
        else:
            z = z_mu

        # generate
        img_rec = decoder(z, reuse=reuse, training=training)

        return z_mu, z_log_sigma_sq, img_rec

    ''' graph '''
    # inputs
    img = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    z_sample = tf.placeholder(tf.float32, shape=[None, z_dim])

    # encode decode
    z_mu, z_log_sigma_sq, img_rec = enc_dec(img, reuse=False, training=True)

    # losses
    rec_loss = tf.reduce_mean(tf.reduce_sum((img - img_rec)**2, axis=[1, 2, 3]))
    kld_loss = -tf.reduce_mean(0.5 * tf.reduce_sum(1 + z_log_sigma_sq - z_mu**2 - tf.exp(z_log_sigma_sq), axis=1))
    loss = rec_loss + kld_loss * 2

    # otpims
    step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(loss)

    # summaries
    summary = utils.summary({rec_loss: 'rec_loss', kld_loss: 'kld_loss'})

    # sample
    _, _, img_rec_sample = enc_dec(img)
    img_sample = decoder(z_sample, training=False)


""" train """
''' init '''
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
summary_writer = tf.summary.FileWriter('./summaries/celeba', sess.graph)

''' initialization '''
ckpt_dir = './checkpoints/celeba'
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

''' train '''
try:
    z_ipt_sample = np.random.normal(size=[100, z_dim])
    img_ipt_sample = data_pool.batch()

    batch_epoch = len(data_pool) // (batch_size)
    max_it = epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # batch data
        img_ipt = data_pool.batch()

        # train D
        summary_opt, _ = sess.run([summary, step], feed_dict={img: img_ipt})
        summary_writer.add_summary(summary_opt, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            img_sample_opt = sess.run(img_sample, feed_dict={z_sample: z_ipt_sample})
            img_rec_sample_opt = sess.run(img_rec_sample, feed_dict={img: img_ipt_sample})

            save_dir = './sample_images_while_training/celeba_generate'
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(img_sample_opt, 10, 10), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))

            save_dir = './sample_images_while_training/celeba_reconstruct'
            utils.mkdir(save_dir + '/')
            n_grid = int(np.ceil(batch_size**0.5))
            img_ori = utils.immerge(img_ipt_sample, n_grid, n_grid)
            img_rec = utils.immerge(img_rec_sample_opt, n_grid, n_grid)
            img_mer = np.concatenate((img_ori, img_rec), 1)
            utils.imwrite(img_mer, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))

except Exception, e:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
