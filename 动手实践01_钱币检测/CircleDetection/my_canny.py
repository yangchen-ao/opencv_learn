import cv2
import numpy as np


class Canny:

    def __init__(self, Guassian_kernal_size, img, HT_high_threshold, HT_low_threshold):
        """
        :param Guassian_kernal_size: 高斯滤波器尺寸
        :param img: 输入的图片，在算法过程中改变
        :param HT_high_threshold: 滞后阈值法中的高阈值
        :param HT_low_threshold: 滞后阈值法中的低阈值
        """
        self.Guassian_kernal_size = Guassian_kernal_size
        self.img = img
        self.h, self.w = img.shape[0:2]
        self.angle = np.zeros([self.h, self.w])
        self.kernal_x = np.array([[-1, 1]])
        self.kernal_y = np.array([[-1], [1]])
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold

    def Get_gradient_img(self):
        """
        计算梯度图和梯度方向矩阵。
        :return: 生成的梯度图
        """

        print('Get_gradient_img')
        # 分别计算xy两方向上的梯度
        new_img_x = np.zeros([self.h, self.w], dtype=np.float)
        new_img_y = np.zeros([self.h, self.w], dtype=np.float)
        for row in range(0, self.h):
            for col in range(0, self.w):
                new_img_y[row][col] = np.sum(
                    np.array([[self.img[row - 1][col]], [self.img[row][col]]]) * self.kernal_y) if row != 0 else 1
                new_img_x[row][col] = np.sum(
                    np.array([self.img[row][col - 1], self.img[row][col]]) * self.kernal_x) if col != 0 else 1
        # 计算全图的梯度大小和方向
        gradient_img, self.angle = cv2.cartToPolar(new_img_x, new_img_y)
        self.angle = np.tan(self.angle)
        self.img = gradient_img.astype(np.uint8)
        return self.img

    def Non_maximum_suppression(self):
        """
        对生成的梯度图进行非极大化抑制，将tan值的大小与正负结合，确定离散中梯度的方向。
        :return: 生成的非极大化抑制结果图
        """

        print('Non_maximum_suppression')
        result = np.zeros([self.h, self.w])
        for row in range(1, self.h - 1):
            for col in range(1, self.w - 1):
                if abs(self.angle[row][col]) > 1:
                    gradient2 = self.img[row - 1][col]
                    gradient4 = self.img[row + 1][col]
                    gradient1 = self.img[row - 1][col - 1] if self.angle[row][col] < 0 else self.img[row - 1][col + 1]
                    gradient3 = self.img[row + 1][col + 1] if self.angle[row][col] < 0 else self.img[row + 1][col - 1]

                else:
                    gradient2 = self.img[row][col - 1]
                    gradient4 = self.img[row][col + 1]
                    gradient1 = self.img[row - 1][col - 1] if self.angle[row][col] < 0 else self.img[row + 1][col - 1]
                    gradient3 = self.img[row + 1][col + 1] if self.angle[row][col] < 0 else self.img[row - 1][col + 1]
                wid = abs(self.angle[row][col]) if abs(self.angle[row][col]) < 1 else 1 / abs(self.angle[row][col])
                temp1 = wid * gradient1 + (1 - wid) * gradient2
                temp2 = wid * gradient3 + (1 - wid) * gradient4
                if self.img[row][col] >= temp1 and self.img[row][col] >= temp2:
                    result[row][col] = self.img[row][col]

        self.img = result
        return self.img

    def Hysteresis_thresholding(self):
        """
        对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，这里的延伸方向为梯度的垂直方向，
        将比低阈值大比高阈值小的点置为高阈值大小，方向在离散点上的确定与非极大化抑制相似。
        :return: 滞后阈值法结果图
        """

        print('Hysteresis_thresholding')
        result = np.zeros((self.h, self.w), np.uint8)
        for row in range(1, self.h - 1):
            for col in range(1, self.w - 1):
                if self.img[row][col] > self.HT_high_threshold:

                    if abs(self.angle[row][col]) < 1:
                        if self.img[row - 1][col] > self.HT_low_threshold:
                            result[row - 1][col] = self.HT_high_threshold
                        if self.img[row + 1][col] > self.HT_low_threshold:
                            result[row + 1][col] = self.HT_high_threshold

                        if self.angle[row][col] > 0:
                            if self.img[row - 1][col - 1] > self.HT_low_threshold:
                                result[row - 1][col - 1] = self.HT_high_threshold
                            if self.img[row + 1][col + 1] > self.HT_low_threshold:
                                result[row + 1][col + 1] = self.HT_high_threshold

                        else:
                            if self.img[row - 1][col + 1] > self.HT_low_threshold:
                                result[row - 1][col + 1] = self.HT_high_threshold
                            if self.img[row + 1][col - 1] > self.HT_low_threshold:
                                result[row + 1][col - 1] = self.HT_high_threshold

                    else:
                        if self.img[row][col - 1] > self.HT_low_threshold:
                            result[row][col - 1] = self.HT_high_threshold
                        if self.img[row][col + 1] > self.HT_low_threshold:
                            result[row][col + 1] = self.HT_high_threshold

                        if self.angle[row][col] > 0:
                            if self.img[row - 1][col - 1] > self.HT_low_threshold:
                                result[row - 1][col - 1] = self.HT_high_threshold
                            if self.img[row + 1][col + 1] > self.HT_low_threshold:
                                result[row + 1][col + 1] = self.HT_high_threshold

                        else:
                            if self.img[row - 1][col + 1] > self.HT_low_threshold:
                                result[row + 1][col - 1] = self.HT_high_threshold
                            if self.img[row + 1][col - 1] > self.HT_low_threshold:
                                result[row + 1][col - 1] = self.HT_high_threshold
        self.img = result
        return self.img

    def canny_algorithm(self):
        """
        按照顺序和步骤调用以上所有成员函数。
        :return: Canny 算法的结果
        """

        self.img = cv2.GaussianBlur(self.img, (self.Guassian_kernal_size, self.Guassian_kernal_size), 0)
        self.Get_gradient_img()
        self.Non_maximum_suppression()
        self.Hysteresis_thresholding()
        return self.img
