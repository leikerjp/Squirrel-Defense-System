import numpy as np
import cv2 as cv

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.nasnet import NASNetMobile, NASNetLarge
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

class_dict = {0:'dog', '1':'horse', 2:"elephant",3:"butterfly",4:"chicken",5:"cat",6:"cow",7:"sheep",8:"spider",9:"squirrel"}

translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}


# Allow memory growth for the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = ResNet50(weights='imagenet')
# model = ResNet50(weights='weights_00.h5')
# model = MobileNetV2(weights='imagenet')
# model = NASNetMobile(weights='imagenet')
# model = NASNetLarge(weights='imagenet')
# model = InceptionV3(weights='imagenet')
# model = tf.keras.models.load_model("model_animal10_00.h5")
# model = tf.keras.models.load_model("lenuts_00.h5")

cap = cv.VideoCapture("C:\\Data\\delme\\squirrel.mp4")
height = 224#331 224
width = 224#331 224
border_width = 5

# Our operations on the frame come here
# gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
backSubtractor = cv.createBackgroundSubtractorMOG2()
backSubtractor.setBackgroundRatio(0.7)  # set to match Matlab
backSubtractor.setNMixtures(3)  # set to match Matlab
backSubtractor.setHistory(1000)  # 500 is default

openStructEle = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
closeStructEle = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))

blobParams = cv.SimpleBlobDetector_Params()
blobParams.minThreshold = 0
blobParams.maxThreshold = 254
blobParams.thresholdStep = 253
blobParams.minDistBetweenBlobs = 50
blobParams.filterByArea = True
blobParams.minArea = (height * width) / 50
blobParams.maxArea = (height * width) / 2  # allow blobs as large as 1/10th the screen to be detected
blobParams.filterByColor = False
blobParams.filterByCircularity = False
blobParams.filterByConvexity = False
blobParams.filterByInertia = False
blobDetector = cv.SimpleBlobDetector_create(blobParams)

frame_count = 0

rolling_window = np.zeros(10)
window_index = 0
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.resize(frame, (width,height))
    frameBord = cv.resize(frame, (width-2*border_width,height-2*border_width))
    frameBord = cv.copyMakeBorder(frameBord, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT,value=[255, 255, 255])

    # back ground subtract
    fgMask = backSubtractor.apply(frameBord)

    # clean up noise
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, openStructEle)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, closeStructEle)

    # threshold out shadows, invert for blob detect
    ret, fgMask = cv.threshold(fgMask, 1, 255, cv.THRESH_BINARY_INV);

    # blob detect
    keypoints = blobDetector.detect(fgMask)

    if (len(keypoints) > 0) and (frame_count % 25 == 0): #50
        # cnts = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # x, y, w, h = cv.boundingRect(cnts[0])
        side = int(np.ceil(keypoints[0].size) * 1.25)
        x = keypoints[0].pt[0]
        y = keypoints[0].pt[1]
        x = int(x - side/2)
        y = int(y - side/2)
        if (x < 0): x = 0
        elif (x > width): x = width
        if (y < 0): y = 0
        elif (y > height): y = height
        ROI = frame[y:y+side, x:x+side]
        ROI = cv.resize(ROI, (width,height))
        # cv.imshow('ROI_{}.png', ROI)

        x = image.img_to_array(ROI)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        print('Predicted:', decode_predictions(preds, top=3)[0])
        rolling_window[window_index % 5] = preds[0][335]
        window_index = window_index + 1
        avg = np.sum(rolling_window)/5
        print('squirrel:\t' + str(preds[0][335]) + "\t\t" + "avg:\t" + str(avg))
        # print(class_dict[preds.argmax()])
        # print(preds)


    # Draw Circles
    frameWithKeypoints = cv.drawKeypoints(frame, keypoints, outImage=np.array([]), color=(0, 0, 255),
                                          flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    maskWithKeypoints = cv.drawKeypoints(fgMask, keypoints, outImage=np.array([]), color=(0, 0, 255),
                                         flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



    # Display the resulting frame
    if ret == True:
        cv.imshow('frame', frameWithKeypoints)
        cv.imshow('mask', maskWithKeypoints)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
        # if cv.waitKey(np.ceil(1000.0 / float(fps)).astype(int)) & 0xFF == ord('q'):
        #     break
    else:
        break

    frame_count = frame_count + 1


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()