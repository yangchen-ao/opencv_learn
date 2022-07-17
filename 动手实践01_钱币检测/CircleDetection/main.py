import cv2
import math

import numpy as np

from my_hough import Hough_transform
from my_canny import Canny

Path = "picture_source/picture.jpg"
Save_Path = "picture_result/"
Reduced_ratio = 2
Guassian_kernal_size = 3
HT_high_threshold = 20
HT_low_threshold = 7
Hough_transform_step = 6
Hough_transform_threshold = 43

if __name__ == '__main__':
    img_gray = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
    img_RGB = cv2.imread(Path)
    y, x = img_gray.shape[0:2]
    img_gray = cv2.resize(img_gray, (int(x / Reduced_ratio), int(y / Reduced_ratio)))
    img_RGB = cv2.resize(img_RGB, (int(x / Reduced_ratio), int(y / Reduced_ratio)))
    # canny takes about 40 seconds
    print('Canny ...')
    canny = Canny(Guassian_kernal_size, img_gray, HT_high_threshold, HT_low_threshold)
    canny.canny_algorithm()
    # 增加图片亮度
    img = canny.img + (255 - np.max(canny.img))
    bright_img = np.where(canny.img > 0, img, 0)

    cv2.imwrite(Save_Path + "canny_result.jpg", bright_img)

    # hough takes about 30 seconds
    print('Hough ...')
    Hough = Hough_transform(canny.img, canny.angle, Hough_transform_step, Hough_transform_threshold)
    circles = Hough.Calculate()
    for circle in circles:
        cv2.circle(img_RGB, (math.ceil(circle[0]), math.ceil(circle[1])), math.ceil(circle[2]), (132, 135, 239), 2)
    cv2.imwrite(Save_Path + "hough_result.jpg", img_RGB)
    print('Finished!')
