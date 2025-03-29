import cv2
import numpy as np
import imutils
import pytesseract

# Set Tesseract OCR path (Change this path based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
image = cv2.imread(r"C:\Users\kaswi\OneDrive\Desktop\download.jpeg")
image = cv2.imread(r"C:\Users\kaswi\OneDrive\Desktop\download.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian Blur
edged = cv2.Canny(gray, 50, 200)  # Edge detection

# Find contours
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

plate = None  # Initialize plate variable

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:  # Looking for a rectangular shape (possible plate)
        plate = approx
        break

if plate is not None:
    x, y, w, h = cv2.boundingRect(plate)
    license_plate = gray[y:y + h, x:x + w]

    # Apply thresholding to improve OCR accuracy
    thresh = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Perform OCR
    text = pytesseract.image_to_string(thresh, config='--psm 8')  # psm 8: Treat image as a single word
    print("Detected License Plate:", text.strip())

    # Draw bounding box on original image
    cv2.drawContours(image, [plate], -1, (0, 255, 0), 2)

    # Display images
    cv2.imshow("Original Image", image)
    cv2.imshow("Edged", edged)
    cv2.imshow("Thresholded Plate", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No license plate detected.")
