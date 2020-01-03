## Author: Tarun Kateja
#Importing relevant packages
import numpy as np
import cv2
import os
import glob
import pytesseract
import codecs, json
import re
import logging
import time

from PIL import Image

#To calculate run time
start_time = time.time()

##Exception logging
logger = logging.getLogger('OCR')
log_file = logging.FileHandler('ocr_errors.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
log_file.setFormatter(formatter)
logger.addHandler(log_file)

##Function for edge detection with optimized parameters (Canny's edge detection algorithm)
def auto_canny(image, sigma = 0.33):
    V = np.median(image)
    lower = int(max(0, (1.0 - sigma) * V))
    upper = int(min(0, (1.0 + sigma) * V))
    edged = cv2.Canny(image, lower, upper, apertureSize=3, L2gradient=True)
    return edged

##Function to convert PDF to TIFF file (images) to perform image processing #This function requires ImageMagick and GhostScript to be installed in system
def pdf_to_image(pdf,img_name, size):
	try:
		if size  <= (4*1024*1024):
			os.system("convert -background white -alpha background -alpha off -density 300 \""+pdf+"\" " +img_name)
		else:
			os.system("convert -background white -alpha background -alpha off -density 190 \""+pdf+"\" " +img_name)
		return "success"
    # except:
    #     logger.error('Could not find the file:%s', master_file_name)
	except Exception as e:
		return "error"

##Function for skew correction (if the images are tilted or lean)
def skew_correction(images):
    rotated_images = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converts image to grey scale
        gray = cv2.bitwise_not(gray) #bitwise not operation
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #Image thresholding to improve image quality
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        rotated_images.append(rotated)
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # print("[INFO] angle: {:.3f}".format(angle))

    return rotated_images

##Function to detect horizontal and vertical lines in the image
#Requires input .tiff file and a parameter for thresholding (window for adaptive thresholding)
#returns bw3(thresholded image, binary masked and joint image)

def detect_lines(image_tiff, c, scale):
    # image = cv2.imread(image_tiff)
    gray = cv2.cvtColor(image_tiff, cv2.COLOR_BGR2GRAY) #convert into gray scale
    edges = auto_canny(gray)

    gray_new = cv2.bitwise_not(gray) #bitwise not operation to get desired image

    bw3 = cv2.adaptiveThreshold(gray_new, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, c, -2)

    horizontal = np.copy(bw3) #copy of thresholded image to identify horizontal and vertical lines
    vertical = np.copy(bw3)
    value = scale
    cols = horizontal.shape[1]
    horizontal_size = int(cols/value)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # cv2.imwrite('horizontal.tiff', horizontal)

    rows = vertical.shape[0]
    vertical_size = int(rows/value)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    # cv2.imwrite('vertical.tiff', vertical)

    masked = cv2.add(horizontal, vertical)
    joints = cv2.bitwise_and(horizontal, vertical)
    # cv2.imwrite('Masked.tiff', masked)

    return bw3, masked, joints

##Funtion to remove duplicate lines
#required after applying hough line transformation as that results multiple hough lines for same edge or lines depending on pixcel intensity

def remove_duplicate_lines(lines):
    length = len(lines)
    # print('Length:',length)
    index = 0
    #for index, line in enumerate(lines):
    while index <= length:
         try:
             line = lines[index]
             # print('Line', line)
             delete = False
             if line[1] == line[3]:
                 i = index+1
                 while i <= length:
                     try:
                        next_line = lines[i]
                        # print('Next Line', next_line)
                        if  next_line[1] == next_line[3]:
                            abs_for_one = abs(line[1]-next_line[1])
                            # print('Diff:', abs_for_one)
                            if abs_for_one < 20:
                                # print('Deleting')
                                delete = True
                                del lines[i]
                            else:
                                i = i+1
                        else:
                            i = i+1
                        # print("i:",i)
                        # print(len(lines))
                     except:
                        logger.error('Could not find next horizontal line for file:%s', master_PDF_file+'_'+filename)
                        break
             elif line[0]==line[2]:
                 # print('Comparing next coordinates')
                 i = index+1
                 while i <= length:
                     try:
                        next_line = lines[i]
                        # print('Next Line', next_line)
                        if  next_line[0] == next_line[2]:
                            abs_for_one = abs(line[0]-next_line[0])
                            # print('Diff:', abs_for_one)
                            if abs_for_one < 20:
                                # print('Deleting')
                                del lines[i]
                                delete = True
                            else:
                                i = i+1
                        else:
                            i = i+1
                     except:
                        logger.error('Could not find next vertical line for file:%s', master_PDF_file+'_'+filename)
                        break
             index = index+1

         except:
             logger.error('list of line is over for file: %s', master_PDF_file+'_'+filename)
             break
    return lines
    print('final lines:', lines)

##Funtion to sort the hough lines required to maintain the structure of data*
def sort_line_list(lines):
    # sort lines into horizontal and vertical
    vertical = []
    horizontal = []
    for line in lines:
        if line[0] == line[2]:
            vertical.append(line)
        elif line[1] == line[3]:
            horizontal.append(line)
    vertical.sort()
    horizontal.sort(key=lambda x: x[1])
    return horizontal, vertical

##Once the table is detected; crops the region of interest (each cell) to perform further OCR
def crop_cells(cropped):
    # image = cv2.imread(cropped)
    try:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        gray_new = cv2.bitwise_not(gray)

        # bw3 = cv2.adaptiveThreshold(gray_new, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        bw3, masked, joints = detect_lines(cropped, 15, 10)
        edges = cv2.Canny(masked, 50, 150, apertureSize=3)
        # edges = auto_canny(masked)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=50)

        lines1 = [list(x[0]) for x in lines]

        # remove duplicates
        lines2 = remove_duplicate_lines(lines1)


        # sort lines into vertical & horizontal lists
        horizontal, vertical = sort_line_list(lines2)
        print(horizontal)
        print(len(horizontal))
        print(vertical)
        print(len(vertical))

        data = []
        # rows = []
        for i, h in enumerate(horizontal):
            if i < len(horizontal) - 1:
                # row = []
                new = []
                for j, v in enumerate(vertical):
                    if i < len(horizontal) - 1 and j < len(vertical) - 1:
                        # every cell before last cell
                        # get width & height
                        width = horizontal[i + 1][1] - h[1]
                        height = vertical[j + 1][0] - v[0]

                    else:
                        # last cell, width = cell start to end of image
                        # get width & height
                        width = tW
                        height = tH
                    tW = width
                    tH = height

                    # get roi (region of interest) to find an x
                    roi = bw3[h[1]:h[1] + width, v[0]:v[0] + height]

                    # save image (for testing)
                    # dir = 'new_max'
                    # if not os.path.exists(dir):
                    #     os.makedirs(dir)
                    # fn = '%s/roi_r%s-c%s.png' % (dir, i, j)
                    img = Image.fromarray(roi)  # numpy open cv array
                    # text = pytesseract.image_to_string(roi, config='--oem 1 -c preserve_interword_spaces=1 -c textord_tabfind_find_tables=1 -c textord_tablefind_recognize_tables=1 -l eng')
                    text1 = pytesseract.image_to_string(img)
                    # print(i, j)
                    new.append(text1)

                    # print (text1)


                    # cv2.imwrite(fn, roi)
                data.append(new)

        return data

    except Exception as e:
        logger.error('Table does not contain cells for file: %s')
        print('error', e)
        return "error"


#Initialization
# C:\AZ\BI Analytics team\Git\OCR_Table_Extraction\data
master_PDF_file = 'C:/AZ/BI Analytics team/Git/OCR_Table_Extraction/data/test.pdf'
pdf_to_image(master_PDF_file, 'image000%d.tiff', 69000)

#time.sleep(10)

data_complete_pdf = []

##Function to take images in sorted manner (Image1, Image 2 and so on)
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

##Image processing starts here
for filename in sorted(glob.glob('image*.tiff'), key=numericalSort):
    print(filename)
    image = cv2.imread(filename)
    bw3, masked, joints = detect_lines(image, 15, 30)
    image2, contours, hierarchy = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data_new = []
    for contour in contours:

        [x, y, w, h] = cv2.boundingRect(contour)
        if h < 50:
            continue

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cropped = image[y: y + h, x: x + w]

        data = crop_cells(cropped)
        data_new.append(data)

    data_complete_pdf.append(data_new)
    # os.remove(filename)

len(data_complete_pdf)

with open('data.txt', 'at+') as f:
    json.dump(data_complete_pdf, f)
    f.close()

print("--- %s seconds ---" % (time.time() - start_time))







