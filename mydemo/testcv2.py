import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\programfiles\Tesseract-OCR\tesseract.exe'

img = cv2.imread('../test/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换到hsv空间

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 形态学结构元

LowerBlue = np.array([90, 190, 100])  # 检测hsv的上下限（蓝色车牌）
UpperBlue = np.array([130, 230, 200])

# inRange 函数将颜色的值设置为 1，如果颜色存在于给定的颜色范围内，则设置为白色，如果颜色不存在于指定的颜色范围内，则设置为 0
mask = cv2.inRange(HSV, LowerBlue, UpperBlue)  # 车牌mask
cv2.imshow('mask', mask)

dilate = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=4)  # 形态学膨胀和开操作把提取的蓝色点连接起来
morph = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel, iterations=6)
cv2.imshow('morph', morph)

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找车牌的轮廓，只找外轮廓就行

# print(len(contours))
img_copy = img.copy()
cv2.drawContours(img_copy, contours, -1, [0, 0, 255], 2)  # 把轮廓画出来
cv2.imshow('img_copy', img_copy)

rect = cv2.boundingRect(contours[0])  # 用矩形把轮廓框出来（轮廓外接矩形）
(x, y, w, h) = rect
cv2.rectangle(img, (x, y), (x + w, y + h), [0, 0, 255], 2)
cv2.imshow('car', img)

roi_img = gray[y:y + h, x:x + w]  # 提取车牌区域进行ocr识别
cv2.imshow('roi_img', roi_img)
# _,roi_thresh = cv2.threshold(roi_img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# open_img = cv2.morphologyEx(roi_thresh,cv2.MORPH_OPEN,kernel)  #适当的形态学操作提高识别率
# cv2.imshow('open_img',open_img)

print('准备输出')
print(pytesseract.image_to_string(roi_img, lang='chi_sim+eng', config='--psm 8 --oem 3'))  # ocr识别

cv2.waitKey(0)
cv2.destroyAllWindows()