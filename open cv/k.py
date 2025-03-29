import cv2
import pytesseract
import numpy as np


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


image_path = r"C:\Users\kaswi\OneDrive\Desktop\download (3).jpeg"
image = cv2.imread(image_path)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blur = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blur, 100, 200)

# 
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

plate = None
for c in contours:
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) == 4:  
        plate = approx
        break

if plate is not None:
    x, y, w, h = cv2.boundingRect(plate)
    plate_img = gray[y:y+h, x:x+w]
    
    # 
    _, thresh = cv2.threshold(plate_img, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 8')
    print("Detected License Plate Number:", text.strip())
    
    #
    cv2.drawContours(image, [plate], -1, (0, 255, 0), 3)
    cv2.putText(image, text.strip(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


cv2.imshow("Detected Plate", image)
cv2.waitKey(0)
cv2.destroyAllWindows()