# Author: Ji Qiu （BUPT）
# cv_xueba@163.com


import numpy as np
import math


class Hough_transform:
    def __init__(self, img, angle, step=5, threshold=135):

        """
        :param img: 输入的图像
        :param angle: 输入的梯度方向矩阵
        :param step: Hough 变换步长大小
        :param threshold: 筛选单元的阈值
        """

        self.img = img
        self.angle = angle
        self.h, self.w = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.h ** 2 + self.w ** 2))
        self.step = step
        self.vote_matrix = np.zeros(
            [math.ceil(self.h / self.step), math.ceil(self.w / self.step), math.ceil(self.radius / self.step)])
        self.threshold = threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        """
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单
        元进行投票。每个点投出来结果为一折线。
        :return:  投票矩阵
        """

        print('Hough_transform_algorithm')

        for row in range(1, self.h - 1):
            for col in range(1, self.w - 1):
                if self.img[row][col] > 0:
                    # 沿着梯度垂直的两个方向
                    b, a, r = row, col, 0
                    while 0 <= b < self.h and 0 <= a < self.w:
                        self.vote_matrix[math.floor(b / self.step)][math.floor(a / self.step)][
                            math.floor(r / self.step)] += 1
                        b = b + self.step * self.angle[row][col]
                        a = a + self.step
                        r = r + math.sqrt((self.step * self.angle[row][col]) ** 2 + self.step ** 2)

                    b = row - self.step * self.angle[row][col]
                    a = col - self.step
                    r = math.sqrt((self.step * self.angle[row][col]) ** 2 + self.step ** 2)
                    while 0 <= b < self.h and 0 <= a < self.w:
                        self.vote_matrix[math.floor(b / self.step)][math.floor(a / self.step)][
                            math.floor(r / self.step)] += 1
                        b = b - self.step * self.angle[row][col]
                        a = a - self.step
                        r = r + math.sqrt((self.step * self.angle[row][col]) ** 2 + self.step ** 2)

        return self.vote_matrix

    def Select_Circle(self):

        """
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制，这里的非极大化抑制我采
        用的是邻近点结果取平均值的方法，而非单纯的取极大值。
        :return: None
        """

        print('Select_Circle')

        maybe = []
        for row in range(0, math.ceil(self.h / self.step)):
            for col in range(0, math.ceil(self.w / self.step)):
                for r in range(0, math.ceil(self.radius / self.step)):
                    if self.vote_matrix[row][col][r] >= self.threshold:
                        b = row * self.step + self.step / 2
                        a = col * self.step + self.step / 2
                        r = r * self.step + self.step / 2
                        maybe.append((math.ceil(a), math.ceil(b), math.ceil(r)))
        if len(maybe) == 0:
            print("No Circle in this threshold.")
            return
        a, b, r = maybe[0]
        possible = []
        for circle in maybe:
            if abs(a - circle[0]) <= 20 and abs(b - circle[1]) <= 20:
                possible.append(circle)
            else:
                result = np.array(possible).mean(axis=0)
                print("Circle core: (%f, %f)  Radius: %f" % (result[0], result[1], result[2]))
                self.circles.append(result)
                possible.clear()
                a, b, r = circle
                possible.append(circle)
        result = np.array(possible).mean(axis=0)
        print("Circle core: (%f, %f)  Radius: %f" % (result[0], result[1], result[2]))
        self.circles.append(result)


    def Calculate(self):
        """
        按照算法顺序调用以上成员函数
        :return: 圆形拟合结果图，圆的坐标及半径集合
        """

        self.Hough_transform_algorithm()
        self.Select_Circle()
        return self.circles
