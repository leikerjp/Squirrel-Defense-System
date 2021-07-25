import numpy as np
import cv2 as cv
import pafy

# cap = cv.VideoCapture("C:\\Users\\jplei\\Workspace\\webcamMotionDetect\\atrium.mp4")
cap = cv.VideoCapture("C:\\Data\\delme\\squirrel.mp4")
# fps = 30
# height = 360
# width = 640
# cap.set(cv.CAP_PROP_FPS, fps)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
# cap = cv.VideoCapture(0)

# fps = 60
height = 256
width = 256
# cap.set(cv.CAP_PROP_FPS, fps)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, width)

# Our operations on the frame come here
# gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
backSubtractor = cv.createBackgroundSubtractorMOG2()
backSubtractor.setBackgroundRatio(0.7)  # set to match Matlab
backSubtractor.setNMixtures(3)  # set to match Matlab
backSubtractor.setHistory(1000)  # 500 is default

openStructEle = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
closeStructEle = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
# openStructEle = cv.getStructuringElement(cv.MORPH_RECT,(10,10))
# closeStructEle = cv.getStructuringElement(cv.MORPH_RECT,(20,20))

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

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.resize(frame, (width,height))
    frame = cv.copyMakeBorder(frame, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT,value=[255, 255, 255])

    # back ground subtract
    fgMask = backSubtractor.apply(frame)

    # clean up noise
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, openStructEle)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, closeStructEle)

    # threshold out shadows, invert for blob detect
    ret, fgMask = cv.threshold(fgMask, 1, 255, cv.THRESH_BINARY_INV);

    # blob detect
    keypoints = blobDetector.detect(fgMask)

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

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()