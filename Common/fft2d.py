import math

import cv2
import numpy as np


def calc_spectrum(complex):
    if complex.ndim == 2:
        dst = abs(complex)
    else:
        dst = cv2.magnitude(complex[:, :, 0], complex[:, :, 1])
    dst = cv2.log(dst + 1)
    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(dst)


def fftshift(img):
    dst = np.zeros(img.shape, img.dtype)
    h, w = dst.shape[:2]
    cy, cx = h // 2, w // 2
    dst[h - cy:, w - cx:] = np.copy(img[0:cy, 0:cx])
    dst[0:cy, 0:cx] = np.copy(img[h - cy:, w - cx:])
    dst[0:cy, w - cx:] = np.copy(img[h - cy:, 0:cx])
    dst[h - cy:, 0:cx] = np.copy(img[0:cy, w - cx:])
    return dst


def exp(knN):
    th = -2 * math.pi * knN
    return complex(math.cos(th), math.sin(th))


def butterfly(pair, L, N, dir):
    for k in range(L):
        Geven, Godd = pair[k], pair[k + L]
        pair[k] = Geven + Godd * exp(dir * k / N)
        pair[k + L] = Geven - Godd * exp(dir * k / N)


def zeropadding(img):
    h, w = img.shape[:2]
    m = 1 << int(np.ceil(np.log2(h)))
    n = 1 << int(np.ceil(np.log2(w)))
    dst = np.zeros((m, n), img.dtype)
    dst[0:h, 0:w] = img[:]
    return dst


def ifft(g):
    fft = parring(g, len(g), -1)
    return [v / len(g) for v in fft]


def fft(g):
    return parring(g, len(g), 1)


def fft2(image):
    pad_img = zeropadding(image)
    tmp = [fft(row) for row in pad_img]
    dst = [fft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)


def ifft2(image):
    tmp = [ifft(row) for row in image]
    dst = [ifft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)


def parring(g, N, dir, start=0, stride=1):
    if N == 1: return [g[start]]
    L = N // 2
    sd = stride * 2
    part1 = parring(g, L, dir, start, sd)
    part2 = parring(g, L, dir, start + stride, sd)
    pair = part1 + part2
    butterfly(pair, L, N, dir)
    return pair


def fft2(image):
    pad_img = zeropadding(image)
    tmp = [fft(row) for row in pad_img]
    dst = [fft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)


def ifft2(image):
    tmp = [ifft(row) for row in image]
    dst = [ifft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)


def FFT(image, mode=2):
    if mode == 1:
        dft = fft2(image)
    elif mode == 2:
        dft = np.fft.fft2(image)
    elif mode == 3:
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = fftshift(dft)
    spectrum = calc_spectrum(dft)
    return dft, spectrum


def IFFT(dft, shape, mode=2):
    dft = fftshift(dft)
    if mode == 1:
        img = ifft2(dft).real
    elif mode == 2:
        img = np.fft.ifft2(dft).real
    elif mode == 3:
        img = cv2.idft(dft, flags=cv2.DFT_SCALE)[:, :, 0]
    img = img[:shape[0], :shape[1]]
    return cv2.convertScaleAbs(img)
