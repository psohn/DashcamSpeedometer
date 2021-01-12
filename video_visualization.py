from optical_flow_utils import *
import numpy as np
import cv2


def play_video(video_name, truth = True):
    '''
    this function takes the name of a video in the data folder and plays it in a new window
    press esc to close it
    '''

    # setting path of video to be read
    video_path = 'data/' + video_name
    cap = cv2.VideoCapture(video_path + '.mp4')
    index = 0
    
    # predicted speed
    speed_predict = pd.read_csv('data/train_pred.csv').iloc[:, 1]
    
    # ground truth
    if truth:
        
        # ground truth speed
        speed_true = np.loadtxt(video_path + '.txt')
        
        # rms error
        speed_error = speed_true - speed_predict
        speed_rms = np.square(speed_error)

    # settings to play video
    while True:
        
        # settings for model to read webcam information
        ret, image = cap.read()
            
        # adding predicted text
        text_font = cv2.FONT_HERSHEY_PLAIN
        text_size = 1
        text_thickness = 1
        cv2.putText(image, 'Predicted Speed: ' + str(round(speed_predict[index], 3)), 
                    (25, 25), text_font, text_size, (0, 0, 255), text_thickness, cv2.LINE_AA)
        
        # adding ground truth text
        if truth:
            
            # ground truth text
            cv2.putText(image, 'True Speed: ' + str(round(speed_true[index], 3)), 
                        (67, 40), text_font, text_size, (0, 0, 255), text_thickness, cv2.LINE_AA)
            
            # rms text
            cv2.putText(image, 'RMS: ' + str(round(speed_rms[index], 1)), 
                        (131, 55), text_font, text_size, (0, 0, 255), text_thickness, cv2.LINE_AA)

        # increasing index
        index += 1
        
        # showing video window
        cv2.imshow('Dashcam', image)

        # escape key set to manually turn off code
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # required to stop video
    cap.release()
    cv2.destroyAllWindows()



def visualize_optical_flow(video_name):
    '''
    function that takes a video and visualizes the dense optical flow
    '''
    

    video_path = 'data/' + video_name
    cap = cv2.VideoCapture(video_path + '.mp4')
    
    ret, frame_1 = cap.read()
    image_1 = preprocess_image(frame_1)

    while(1):
        # settings for model to read webcam information
        ret, image = cap.read()
        image_2 = preprocess_image(image)

        image_hsv = optical_flow(image_1, image_2)
        
        cv2.imshow('Dashcam', image_hsv[..., 2])

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
        image_1 = image_2

    cap.release()
    cv2.destroyAllWindows()