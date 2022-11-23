import numpy as np, cv2
from Common.dft2d import exp, calc_spectrum, fftshift
from Common.dft2d import zeropadding


def butterfly(pair, L, N, dir):
    for k in range(L):
        Geven, Godd = pair[k], pair[k + L]
        pair[k] = Geven + Godd * exp(dir * k / N)
        pair[k + L] = Geven - Godd * exp(dir * k / N)


def parring(g, N, dir, start=0, stride=1):
    if N == 1: return [g[start]]
    L = N // 2
    sd = stride * 2
    part1 = parring(g, L, dir, start, sd)
    part2 = parring(g, L, dir, start + stride, sd)
    pair = part1 + part2
    butterfly(pair, L, N, dir)
    return pair


def fft(g):
    return parring(g, len(g), 1)


def ifft(g):
    fft = parring(g, len(g), -1)
    return [v / len(g) for v in fft]


def fft2(image):
    pad_img = zeropadding(image)
    tmp = [fft(row) for row in pad_img]
    dst = [fft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)


def ifft2(image):
    tmp = [ifft(row) for row in image]
    dst = [ifft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)


image = cv2.imread("images-5/dft_240.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

dft1 = fft2(image)
dft2 = np.fft.fft2(image)
dft3 = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

spectrum1 = calc_spectrum(fftshift(dft1))
spectrum2 = calc_spectrum(fftshift(dft2))
spectrum3 = calc_spectrum(fftshift(dft3))

idft1 = fft2(dft1).real
idft2 = np.fft.ifft2(dft1).real
idft3 = cv2.idft(dft3, flags=cv2.DFT_SCALE)[:, :, 0]

print("user 방법 변환 행렬 크기:", dft1.shape)
print("np.fft 방법 변환 행렬 크기:", dft2.shape)
print("cv2.dft 방법 변환 행렬 크기:", dft3.shape)

cv2.imshow("spectrum1", spectrum1)
cv2.imshow("spectrum2-np.fft", spectrum2)
cv2.imshow("spectrum3-OpenCV", spectrum3)
cv2.imshow("idft_img1", cv2.convertScaleAbs(idft1))
cv2.imshow("idft_img2", cv2.convertScaleAbs(idft2))
cv2.imshow("idft_img3", cv2.convertScaleAbs(idft3))
cv2.waitKey(0)
