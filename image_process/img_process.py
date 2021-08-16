import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def ImageBlurred():
    img = cv.imread('./back_gas_production1_2561.jpg')
    #blur = cv.blur(img, (5, 5))
    #高斯模糊处理
    #blur = cv.GaussianBlur(img, (3, 3), 0)
    blur = cv.bilateralFilter(img, 9, 150, 150)
    cv.imwrite('./output.jpg', blur)
    blur = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

def ImageIncreaseBrightness():
    # 调整彩色图像的亮度
    img = cv.imread('back_gas_production1_2561.jpg')
    img_t = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_t)
    # 增加图像亮度
    v1 = np.clip(cv.add(1 * v, 30), 0, 255)
    img1 = np.uint8(cv.merge((h, s, v1)))
    img1 = cv.cvtColor(img1, cv.COLOR_HSV2BGR)
    cv.imwrite('brightness.jpg', img1)

def ImageIncreaseContrast():
    # 调整彩色图像的对比度
    img = cv.imread('back_gas_production1_2561.jpg')
    img_t = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_t)

    # 增加图像对比度
    v2 = np.clip(cv.add(0.8 * v, 60), 0, 255)
    v2 = np.uint8(v2)
    img2 = np.uint8(cv.merge((h, s, v2)))
    img2 = cv.cvtColor(img2, cv.COLOR_HSV2BGR)
    tmp = np.hstack((img, img2))
    cv.imwrite('contrast.jpg', tmp)

#增加亮度和对比度方法二, 速度太慢
def ImageIncreaseBrightnessAndContrast_2():
    print('start...')
    mat = cv.imread('back_gas_production1_2561.jpg')
    rows, cols, channels = mat.shape
    print(rows, cols, channels)
    dst = mat.copy()
    a = 1.2#控制对比度
    b = 80#控制亮度
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                color = mat[i, j][c]*a + b
                if color > 255:
                    dst[i, j][c] = 255
                elif color < 0:
                    dst[i, j][c] = 0
                else:
                    dst[i, j][c] = color
    #out_mat = np.hstack(mat, dst)
    cv.imwrite('path2.jpg', dst)
#path 3
"""
函数详解：
一共有七个参数：前4个是两张要合成的图片及它们所占比例，
                            第5个double gamma起微调作用，
                            第6个OutputArray dst是合成后的图片，
                            第7个输出的图片的类型（可选参数，默认-1）
两个图片加成输出的图片为：dst = src1 * alpha+src2 * beta + gamma
addWeighted(InputArray_src1, 
            double_alpha, 
            InputArray_src2, 
            double_beta, 
            double_gamma, 
            OutputArray_dst, 
            int_dtype=-1
            );
"""
def ImageIncreaseBrightnessAndContrast_3():
    mat = cv.imread('back_gas_production1_2561.jpg')
    rows, cols, channels = mat.shape
    blank_mat = np.zeros([rows, cols, channels], mat.dtype)
    a = 1.5
    b = 0
    dst = cv.addWeighted(mat, a, blank_mat, 1-a, b)
    temp = np.hstack((mat, dst))
    cv.imwrite('path_3.jpg', temp)

#增加亮度和对比度方法四
def ImageIncreaseBrightnessAndContrast_4():
    mat = cv.imread('back_gas_production1_2561.jpg')
    a = 1.5
    b = 20
    dst = np.uint8(np.clip((mat*a + b), 0, 255))
    temp = np.hstack((mat, dst))
    cv.imwrite('path_4.jpg', temp)


#图像直方图增加图片的清晰度
# 彩色图像全局直方图均衡化
def hisEqulColor1():
    mat = cv.imread('back_gas_production1_2561.jpg')
    img = mat.copy()
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv.split(ycrcb)
    # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    cv.equalizeHist(channels[0], channels[0])
    # 将处理后的通道和没有处理的两个通道合并，命名为ycrcb
    cv.merge(channels, ycrcb)
    # 将YCrCb图像转换回RGB图像
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
    temp = np.hstack((mat, img))
    cv.imwrite('hisEqulColor1.jpg', temp)

# 彩色图像进行自适应直方图均衡化，代码同上的地方不再添加注释
def hisEqulColor2():
    mat = cv.imread('back_gas_production1_2561.jpg')
    img = mat.copy()
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
    temp = np.hstack((mat, img))
    cv.imwrite('hisequlcolor2.jpg', temp)

# histogram normalization
def hist_normalization(a=0, b=255):
    # histogram normalization
    img = cv.imread('back_gas_production1_2561.jpg')
    # get max and min
    c = img.min()
    d = img.max()
    print(c, d)
    out = img.copy()
    # normalization
    out = (b - a) / (d - c) * (out - c) + a
    out[out < a] = a
    out[out > b] = b
    out = out.astype(np.uint8)
    # Display histogram
    plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig("out_his.jpg")
    plt.show()
    # Save result
    cv.imshow("result", out)
    cv.imwrite("out.jpg", out)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    print('debug start...')
    # ImageIncreaseBrightness()
    #ImageIncreaseContrast()
    # ImageIncreaseBrightnessAndContrast_4()
    #ImageBlurred()
    # hisEqulColor2()
    hist_normalization()