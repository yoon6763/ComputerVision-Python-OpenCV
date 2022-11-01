import numpy as np, cv2
from Common.interplation import scaling_nearest


def bilinear_value(img, pt):
    x, y = np.int32(pt)
    if x >= img.shape[1] - 1: x = x - 1
    if y >= img.shape[0] - 1: y = y - 1

    P1, P2, P3, P4 = np.float32(img[y:y + 2, x:x + 2].flatten())

    alpha, beta = pt[1] - y, pt[0] - x
    M1 = P1 + alpha * (P3 - P1)
    M2 = P2 + alpha * (P4 - P2)
    P = M1 + beta * (M2 - M1)
    return np.clip(P, 0, 255)


def scaling_bilinear(img, size):
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])

    dst = [[bilinear_value(img, (j / ratioX, i / ratioY))
            for j in range(size[0])]
           for i in range(size[1])]
    return np.array(dst, img.dtype)


image = cv2.imread("images/interpolation.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 에러")

size = (350, 400)
dst1 = scaling_bilinear(image, size)
dst2 = scaling_nearest(image, size)
dst3 = cv2.resize(image, size, 0, 0, cv2.INTER_LINEAR)
dst4 = cv2.resize(image, size, 0, 0, cv2.INTER_NEAREST)

cv2.imshow("image", image)
cv2.imshow("User_bilinear", dst1)
cv2.imshow("User_Nearest", dst2)
cv2.imshow("OpenCV_bilinear", dst3)
cv2.imshow("OpenCV_Nearest", dst4)
cv2.waitKey(0)
