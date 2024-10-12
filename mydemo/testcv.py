import cv2
import numpy as np
import pytesseract


print('begin')
# 创建级联分类器
car = cv2.CascadeClassifier('./haarcascade_russian_plate_number.xml')

img = cv2.imread('./car3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 形态学结构元

car_nums = car.detectMultiScale(gray)  # 车牌检测（检测出来的框偏大）
print(car_nums)
for car_num in car_nums:
    (x, y, w, h) = car_num
    cv2.rectangle(img, (x, y), (x + w, y + h), [0, 0, 255], 2)  # 用矩形把车牌框起来
    roi = gray[y:y + h, x:x + w]  # 把车牌图片提取出来

    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 对提取的车牌二值化

    open_img = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel)  # 形态学开操作（二值化后效果还是差点）

    cv2.imshow('open_img', open_img)

    print(pytesseract.image_to_string(open_img, lang='chi_sim+eng', config='--psm 7 --oem 3'))  # 进行ocr识别

#cv2.imshow('car', img)

cv2.waitKey(0)
cv2.destroyAllWindows()