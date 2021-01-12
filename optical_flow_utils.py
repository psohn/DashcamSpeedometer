# importing libraries
import cv2
import numpy as np
import skvideo.io as sk
from sklearn.model_selection import train_test_split
import pandas as pd



def apply_gray(image):
    '''
    function used to take an image and convert it to grayscale
    '''

    # applying grayscale and gaussian blur
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image_gray



def apply_yw_filter(image):
    '''
    function used to take an image and filter out nonwhite and nonyellow
    '''
    
    # upping the brightness
    image_gray = apply_gray(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # yellow ranges
    lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
    upper_yellow = np.array([255, 255, 255], dtype = 'uint8')
    
    # creating masks
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(image_gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_yellow, mask_white)
    
    # applying masks
    image_mask_yw = cv2.bitwise_and(image_gray, mask_yw)
    
    return image_mask_yw



def canny_edge(image):
    '''
    function used to take an image and find the canny edges
    '''
    
    # finding canny edges
    image_edge = cv2.Canny(image, 100, 300)
    
    return image_edge



def apply_mask(image):
    '''
    function used to take an image and apply a mask to it
    '''
    
    # applying a polygon mask
    mask = np.zeros_like(image)
    mask_coordinates = [[0, 350], [640, 350], [640, 280], [415, 240], [210, 240], [0, 280]]
    cv2.fillConvexPoly(mask, np.array(mask_coordinates), (1, 1, 1))
    image_mask = image * mask
    
    return image_mask



def apply_brightness_contrast(image, brightness = 0, contrast = 0):
    '''
    function used to take an image and change the brightness and contrast
    '''

    if brightness != 0:
        
        if brightness > 0:
            shadow = brightness
            highlight = 255
            
        else:
            
            shadow = 0
            highlight = 255 + brightness
            
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
        
    else:
        
        buf = image.copy()

    if contrast != 0:
        
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
    


def optical_flow(image_1, image_2):
    '''
    function that takes two consecutive frames and calculates its optical flow
    '''
    
    # optical flow calculation
    flow = cv2.calcOpticalFlowFarneback(image_1, image_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # creating hsv optical flow image
    hsv_image = np.empty(shape = (image_1.shape[0], image_1.shape[1], 3))
    hsv_image[..., 0] = ang * 180 / np.pi / 2
    hsv_image[..., 1] = 255
    hsv_image[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    return hsv_image



def preprocess_image(image):
    '''
    function used to take an image and find the shi-tomasi corners
    '''
    
    # apply brightness and contrast
    image_bright = apply_brightness_contrast(image, 125, 50)
    
    # yellow and white filter
    image_yw = apply_yw_filter(image_bright)
    
    # gaussian blur
    image_blur = cv2.GaussianBlur(image_yw, (5, 5), 0)
    
    # canny edge
    image_canny = canny_edge(image_blur)
    
    # apply mask
    image_mask = apply_mask(image_canny)
    
    # combine original image and edges
    image_gray = apply_gray(image)
    image_final = cv2.add(image_gray, image_mask)
    
    return image_final



def preprocess_video(file_name):
    '''
    this function takes a video and takes its consecutive frames to calculate its dense optical flow
    '''
    
    # reading data
    print('---loading data---')
    path = 'data/' + file_name
    video = sk.vread(path + '.mp4')
    print('---data loaded---')
    
    # extracting features of video
    width = video.shape[1]
    length = video.shape[2]
    frame_num = video.shape[0]

    # creating empty array
    video_preprocess = np.empty(shape = (frame_num - 1, width, length))
    image_1 = np.empty(shape = (width, length))
    
    # calculating optical flow of consecutive frames and saving them
    print('---calculating optical flow---')
    for index, frame in enumerate(video):
        
        image_2 = preprocess_image(frame)
        
        if index != 0:
            
            if index%1000 == 0:
                
                print('---' + str(index) + '/' + str(frame_num) + ' calculated---')
            
            image_hsv = optical_flow(image_1, image_2)
            video_preprocess[index - 1] = image_hsv[..., 2]
            image_1 = image_2
        
        else: 
            
            image_1 = image_2
    print('---' + str(frame_num) + '/' + str(frame_num) + ' calculated---')
    print('---done processing video---')
    
    return video_preprocess.reshape(frame_num - 1, width, length, 1)