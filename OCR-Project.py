import pytesseract
import cv2
import numpy as np
from PIL import Image

def pre_processing():
    # read the image
    img = cv2.imread("plate_eng.jpg")

    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # filter and blur
    filter = cv2.bilateralFilter(bin, 11, 17, 17)
    blur = cv2.medianBlur(filter, 5)

    # find edges
    Edges = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(img, Edges, -1, (0, 255, 0), 3)

    #confirm the countours, aproximate the countours and create a cropped image
    for countour in Edges:
        perimeter = cv2.arcLength(countour, True)

        if perimeter > 500:
            aprox = cv2.approxPolyDP(countour, 0.02 * perimeter, True)

            if len(aprox) == 4:
                (x, y, w, h) = cv2.boundingRect(countour)
                cv2.rectangle(img, (x, y ), (x + w, y + h), (0, 255, 0), 2)
                new_img = img[y:y + h, x:x + w]
                cv2.imwrite("cropped_img.jpg", new_img)
                img_resize = cv2.resize(new_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


    # depedends of the language of the text, use eng for english, chi_sim for simplified chinese, etc.
    text = pytesseract.image_to_string(new_img, lang='eng')
    print(text)
    if text == '':
        print("No text found")

    cv2.imshow("Countours", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

pre_processing()