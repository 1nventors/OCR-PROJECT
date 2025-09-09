# OCR Text Recognition App 📝🔍  (WORKING ON)

The goal was to build an application that allows users to **recognize text from images** using **OCR (Optical Character Recognition)** powered by **Tesseract** and **OpenCV**.  
It supports multiple languages and can be applied to documents, license plates, or general images containing text.  

---

## 📌 Features  

- **Preprocessing of images** for better OCR results:  
  - Grayscale conversion  
  - Thresholding  
  - Filtering and blurring  
  - Contour detection with bounding boxes  
- **Text recognition** with `pytesseract`.  
- **Language support**: English (`eng`), Portuguese (`por`), Simplified Chinese (`chi_sim`).  
- Works with different types of input images:  
  - License plates  
  - Documents  
  - General text images  

---

## 🛠️ Technologies Used  

- **Python** (main language)  
- **OpenCV** (image processing)  
- **pytesseract / Tesseract OCR** (optical character recognition)  
- **Pillow** (image handling)  
- **NumPy** (matrix and array operations)  
