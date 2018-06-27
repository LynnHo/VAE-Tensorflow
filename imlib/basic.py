from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imlib.dtype import im2float
import numpy as np
import skimage.io as iio


def imread(path, as_gray=False):
    """Read image.

    Returns:
        Float64 image in [-1.0, 1.0].
    """
    image = iio.imread(path, as_gray)
    if image.dtype == np.uint8:
        image = image / 127.5 - 1
    return image


def imwrite(image, path):
    """Save an [-1.0, 1.0] image."""
    iio.imsave(path, im2float(image))


def imshow(image):
    """Show a [-1.0, 1.0] image."""
    iio.imshow(im2float(image))


show = iio.show
