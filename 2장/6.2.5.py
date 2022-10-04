import numpy as np, cv2

image = cv2.imread("C:/Users/yoon9/Downloads/images/contrast.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

noimage = np.zeros(image.shape[:2], image.dtype)
avg = cv2.mean(image)[0] / 2.0

dst1 = cv2.scaleAdd(image, 0.5, noimage)
dst2 = cv2.scaleAdd(image, 2.0, noimage)
dst3 = cv2.addWeighted(image, 0.5, noimage, 0, avg)
dst4 = cv2.addWeighted(image, 2.0, noimage, 0, -avg)

cv2.imshow("image", image)
cv2.imshow("dst1 - decrease contrast", dst1)
cv2.imshow("dst2 - increase contrast", dst2)
cv2.imshow("dst3 - decrease contrast using average", dst3)
cv2.imshow("dst4 - increase contrast using avergae", dst4)
cv2.waitKey(0)