"""
solo la parte di randaugment semplificata... vedi
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
"""
import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance, ImageChops
import PIL
import numpy as np
import torchvision.transforms as transforms
import torch
_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_PIL_VER
_FILL = (128, 128, 128)

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.
_HPARAMS_DEFAULT = dict(
    cutout=8,
    translate_const=250,
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)

def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation

def _check_args_tf(kwargs):
    kwargs['resample'] = _interpolation(kwargs)

def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)

def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)

def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)

def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)

def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    return img.rotate(degrees, **kwargs)

def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)

def invert(img, **__):
    return ImageOps.invert(img)

def cutout(img, length):
    trans = transforms.ToPILImage()
    trans1 = transforms.ToTensor()
    img = trans1(img)
    h = img.size(1)
    w = img.size(2)

    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)
    mask[y1: y2, x1: x2] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask

    return trans(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, hparams):
    # range [0.1, 1.9]
    return (level / _MAX_LEVEL) * 1.8 + 0.1,


def _enhance_increasing_level_to_arg(level, hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9]
    level = (level / _MAX_LEVEL) * .9
    level = 1.0 + _randomly_negate(level)
    return level,


def _shear_level_to_arg(level, hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return level,


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return level,


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get('translate_pct', 0.45)
    level = (level / _MAX_LEVEL) * translate_pct
    level = _randomly_negate(level)
    return level,


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return int((level / _MAX_LEVEL) * 4),


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return 4 - _posterize_level_to_arg(level, hparams)[0],


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return int((level / _MAX_LEVEL) * 4) + 4,


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return int((level / _MAX_LEVEL) * 256),


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return 256 - _solarize_level_to_arg(level, _hparams)[0],


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return int((level / _MAX_LEVEL) * 110),


LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Cutout':None,
    'Rotate': _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    'Posterize': _posterize_level_to_arg,
    'PosterizeIncreasing': _posterize_increasing_level_to_arg,
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeIncreasing': _solarize_increasing_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'ColorIncreasing': _enhance_increasing_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'ContrastIncreasing': _enhance_increasing_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'BrightnessIncreasing': _enhance_increasing_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'SharpnessIncreasing': _enhance_increasing_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'PosterizeIncreasing': posterize,
    'PosterizeOriginal': posterize,
    'Solarize': solarize,
    'SolarizeIncreasing': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
    'Cutout': cutout
}

_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    'Cutout'
]

class RandAugment:
    def __init__(self, num_layers, magnitude, cutout_size = 8):
        self.ops = [AugmentOp(name, prob=0.5, magnitude=magnitude, cutout_size=cutout_size) for name in _RAND_TRANSFORMS]
        self.num_layers = num_layers

    def __call__(self, img):
        ops = np.random.choice(self.ops, self.num_layers, replace=False)
        for op in ops:
            img = op(img)
        return img


class AugmentOp:

    def __init__(self, name, magnitude, cutout_size=8, prob=0.5):
        hparams = _HPARAMS_DEFAULT
        self.name = name
        self.cutout_size = cutout_size
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = _HPARAMS_DEFAULT
        self.kwargs = dict(fillcolor=_FILL, resample= _RANDOM_INTERPOLATION,)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        if self.name == 'Cutout':
            return cutout(img, self.cutout_size)
        else:
            magnitude = max(0, self.magnitude)
            level_args = self.level_fn(self.magnitude, self.hparams) if self.level_fn is not None else tuple()
            return self.aug_fn(img, *level_args, **self.kwargs)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
