import cv2

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    '''
    zip(*)相当于解压, 是zip的逆过程
    key=lambda b: b[1][i]中, b只是一个变量名, 可以是任何一个变量
    b[1][i]指按第二元素中的第i个元素就行排序, 也就是按boundingBoxes中的第i个元素进行排序
    '''
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized