# 导入工具包
import cv2
import myutils
import argparse
import numpy as np
from imutils import contours

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-t", "--template", required=True,
                help="path to template ORC-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'此处开始对模板进行处理'
# 读取一个模板图像
img = cv2.imread(args["template"])
# cv_show('img', img)

# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show('ref', ref)

# 二值图像
# [1]表示返回第二个参数
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
# cv_show('ref', ref)

# 计算轮廓
# cv2.RETR_EXTERNAL只检测外轮廓, cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]
# cv_show('img', img)

# 遍历每一个轮廓
# enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列
digits = dict()
for (i, c) in enumerate(refCnts):
    # 计算外界矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)

    # 从ref中裁剪图像
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))

    # 每一个数字对应一个模板
    digits[i+1] = roi


'此处开始对银行卡图片进行处理'
# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 获取输入图像, 预处理
image = cv2.imread(args["image"])
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show('gray', gray)

# 礼帽操作, 突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
# cv_show('tophat', tophat)

# 边界提取
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
gradY = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
gradX = cv2.convertScaleAbs(gradX)
gradY = cv2.convertScaleAbs(gradY)
gradXY = cv2.add(gradX,gradY)
# cv_show('gradXY', gradXY)

# 第一个闭操作, 将数字连在一起
gradXY = cv2.morphologyEx(gradXY, cv2.MORPH_CLOSE, rectKernel)
# cv_show('gradXY', gradXY)

# cv2.THRESH_OTSU会自动寻找合适的阈值
# 适合双峰, 需要把阈值参数设置为0
thresh = cv2.threshold(gradXY, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv_show('thresh', thresh)

# 第二个闭操作, 填充目标区域的空隙
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
# cv_show('thresh', thresh)

# 计算轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(),
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
# cv_show('cur_img', cur_img)

# 遍历轮廓
locs = list()
selectd_cnts = list()
for (i, c) in enumerate(cnts):
    # 遍历矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 选择合适的区域, 根据实际任务调整
    if 2.5 < ar < 4.0:
        if (40 < w < 55) and (10 < h < 20):
            # 留下符合的轮廓
            locs.append((x, y, w, h))
            selectd_cnts.append(c)

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])

now_img = image.copy()
cv2.drawContours(now_img, selectd_cnts, -1, (0, 0, 255), 3)
# cv_show('now_img', now_img)

'此处开始将模板数字轮廓和银行卡数字轮廓进行匹配'
# 遍历每一个轮廓中的数字
output = list()
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # 保存最终匹配的数字
    groupOutput = list()

    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    # cv_show('group', group)

    # 预处理
    group = cv2.threshold(group, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv_show('group', group)

    # 计算每一组的轮廓
    digitsCnts, hierarchy = cv2.findContours(group.copy(),
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitsCnts = contours.sort_contours(digitsCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitsCnts:
        # 找到当前数值的轮廓, resize成合适的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # 在模板中计算每一个数值的得分
        scores = []
        for (digit, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

    # 在图片中画出识别的数字
    cv2.rectangle(image, (gX - 5, gY - 5),
                  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv_show("Image", image)
