import pytesseract
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r'D:\programfiles\Tesseract-OCR\tesseract.exe'


file=r'../test/1.jpg'
image=Image.open(file)
print(pytesseract.image_to_string(image,lang='chi_sim+eng'))