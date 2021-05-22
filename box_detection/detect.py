import cv2
import os
import numpy as np
import pandas as pd

def get_boxes(image_path, lw, threshold_min, cc, mode, output_path):
    """
    Get the check boxes in an image
    
    Arguments:
        image_path = path of the image file
        lw = length of the kernel
        threshold_min = min value for binarization
        cc = connectivity for the pixel separation
        mode = either "table" or "check boxes"
        output_path = saving image path
    """
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    ## binarization
    ret, img_bin = cv2.threshold(img, threshold_min, 240, cv2.THRESH_BINARY_INV)

    ## kernels
    lin_width = lw
    k_h = np.ones((1, lin_width), np.uint8)
    k_v = np.ones((lin_width, 1), np.uint8)

    ## image_morphing
    img_dil = img_bin.copy()
    img_bin_h = cv2.morphologyEx(img_dil, cv2.MORPH_OPEN, k_h)
    img_bin_v = cv2.morphologyEx(img_dil, cv2.MORPH_OPEN, k_v)
    img_mer = img_bin_h|img_bin_v
    
    # connected components
    args = {}
    args["connectivity"] = cc
    cc_im, labels, stats, _ = cv2.connectedComponentsWithStats(img_mer, args["connectivity"], cv2.CV_32S)
    
    # getting the bounding boxes and areas
    b_boxes = []
    b_area = []
    for x, y, w, h, area in stats:
        b_boxes.append([x, y, w, h])
        b_area.append(area)
    max_ar = max(b_area)
    min_ar = min(b_area)
    range_ar = max_ar - min_ar
    b_area = [round(x/(range_ar), 3) for x in b_area]    
        
    # final processing
    out_image = cv2.imread(image_path)
    output_list = []
    for i,j in enumerate(b_boxes):
        x = j[0]
        y = j[1]
        w = j[2]
        h = j[3]
        
        out = {}
        out['xmin'] = x
        out['ymin'] = y
        out['width'] = w
        out['height'] = h
        
        if mode == "check boxes" and abs(w-h) < 1.4e-10: 
            cv2.rectangle(out_image, (x, y), (x+w, y+h), (0,255,0), 2)
            output_list.append(out)
        
        elif b_area[i] > .005:
            cv2.rectangle(out_image, (x, y), (x+w, y+h), (0,255,0), 2)
            output_list.append(out)
        
    print("Saving final image to the output_path")    
    cv2.imwrite(output_path, out_image)
    print("Image saved")
    return pd.DataFrame(output_list)
