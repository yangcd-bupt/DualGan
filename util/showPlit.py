import numpy as np
import cv2


def showplt(X, name):
    array1 = X[0].numpy()  # 将tensor数据转为numpy数据
    maxValue = array1.max()
    array1 = array1 * 255 / maxValue  # normalize，将图像数据扩展到[0,255]
    mat = np.uint8(array1)  # float32-->uint8
    mat = mat.transpose(1, 2, 0)  # mat_shape: (982, 814，3)
    cv2.imshow(name, mat)
