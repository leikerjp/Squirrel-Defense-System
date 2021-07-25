import numpy as np
import cv2 as cv

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from MotionTracker import MotionTracker

# Allow memory growth for the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = ResNet50(weights='imagenet')

cap = cv.VideoCapture("C:\\Data\\delme\\squirrel.mp4")

# 360p
resolution = (640,360) # 480,360
resnet_resolution = (224,224)
center = (resolution[0]/2,resolution[1]/2)
framerate = 50.0
border_width = 5

background_subtractor = cv.createBackgroundSubtractorMOG2()
background_subtractor.setBackgroundRatio(0.7)  # set to match Matlab
background_subtractor.setNMixtures(3)  # set to match Matlab
background_subtractor.setHistory(1000)  # 500 is default

blob_params = cv.SimpleBlobDetector_Params()
blob_params.minThreshold = 0
blob_params.maxThreshold = 254
blob_params.thresholdStep = 253
blob_params.minDistBetweenBlobs = 50
blob_params.filterByArea = True
blob_params.minArea = (resolution[0] * resolution[1]) / 50 # Manually tuned
blob_params.maxArea = (resolution[0] * resolution[1]) / 1.5  # Manually tuned
blob_params.filterByColor = False
blob_params.filterByCircularity = False
blob_params.filterByConvexity = False
blob_params.filterByInertia = False
blob_detector = cv.SimpleBlobDetector_create(blob_params)

open_element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
close_element = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
mTracker = MotionTracker(height=resolution[1],
                         width=resolution[0],
                         fps=framerate,
                         background_subtractor=background_subtractor,
                         blob_detector=blob_detector,
                         open_element=open_element,
                         close_element=close_element)

frame_count = 0

rolling_window = np.zeros(10)
window_index = 0

cap = cv.VideoCapture("C:\\Data\\delme\\squirrel.mp4")
# cap = cv.VideoCapture("atrium.mp4")

text = ''
text_location = (0,0)

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.resize(frame, resolution)
    frameBord = cv.resize(frame, (resolution[0]-2*border_width,resolution[1]-2*border_width))
    frameBord = cv.copyMakeBorder(frameBord, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT,value=[255, 255, 255])

    if ret == False:
        print('Restarting Video')
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    else:
        tracks = mTracker.detect_and_track(frameBord)
        tracks.sort(key=lambda track: track.id)
        keypoints = [track.centroid for track in tracks]

        if len(keypoints) > 0:
            x_move = -1 * (keypoints[0].pt[0] - center[0])
            y_move = -1 * (keypoints[0].pt[1] - center[1])
            trans = np.float32([[1,0,x_move],[0,1,y_move]])
            frame = cv.warpAffine(frame, trans, resolution)

            for keypoint in keypoints:
                keypoint.pt = (keypoint.pt[0] + x_move, keypoint.pt[1] + y_move)

            if(frame_count % 25 == 0):
                side = int(np.ceil(keypoints[0].size) * 1.25)
                x = keypoints[0].pt[0]
                y = keypoints[0].pt[1]
                x = int(x - side/2)
                y = int(y - side/2)
                if (x < 0): x = 0
                elif (x > resnet_resolution[0]): x = resnet_resolution[0]
                if (y < 0): y = 0
                elif (y > resnet_resolution[1]): y = resnet_resolution[1]
                ROI = frame[y:y+side, x:x+side]
                ROI = cv.resize(ROI, resnet_resolution)

                x = image.img_to_array(ROI)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                preds = model.predict(x)
                decoded_preds = decode_predictions(preds, top=1)[0]
                text = decoded_preds[0][1]
                text_location = (int(keypoints[0].pt[0]), int(keypoints[0].pt[1] + 50))


                rolling_window[window_index % 5] = preds[0][335]
                window_index = window_index + 1
                avg = np.sum(rolling_window) / 5
                print('squirrel:\t' + str(preds[0][335]) + "\t\t" + "avg:\t" + str(avg))



        frame = cv.drawKeypoints(frame, keypoints, outImage=np.array([]), color=(0, 0, 255),
                                       flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        frame = cv.putText(frame, text, text_location,
                           cv.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2, cv.LINE_AA)

        cv.imshow('Frame', frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break


    frame_count = frame_count + 1


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()