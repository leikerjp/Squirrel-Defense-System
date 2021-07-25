import cv2 as cv
import numpy as np

from MotionTracker import MotionTracker


# 360p
resolution = (640,360) # 480,360
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


cap = cv.VideoCapture("C:\\Data\\delme\\squirrel.mp4")
# cap = cv.VideoCapture("atrium.mp4")
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


        frame = cv.drawKeypoints(frame, keypoints, outImage=np.array([]), color=(0, 0, 255),
                                       flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv.imshow('Frame', frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break


# while (cap.isOpened()):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     if ret == False:
#         print('Restarting Video')
#         cap.set(cv.CAP_PROP_POS_FRAMES, 0)
#     else:
#         # keypoints = mTracker.detect_and_track(frame)
#         #print(index)
#
#         # Run motion detector on frame
#         mTracker._detect(frame)
#         # print("detected centroids\t", end='')
#         # for centroid in mTracker.detected_keypoints:
#         #     print(centroid.pt, end='')
#         # print('')
#
#         # Create centroid predictions of for each track
#         mTracker._predictNewLocationsOfTracks()
#
#
#         keypoints = [track.centroid for track in mTracker.tracks]
#         frameWithKeypoints = cv.drawKeypoints(frame, keypoints, outImage=np.array([]), color=(0, 255, 255),
#                                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         mTracker.fg = cv.drawKeypoints(mTracker.fg, keypoints, outImage=np.array([]), color=(0, 255, 255),
#                                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#         # print("track centroids\t", end='')
#         # for keypoint in keypoints:
#         #     print(str(keypoint.pt) + '\t', end='')
#         # print('')
#
#         # Assign detections to tracks / create new tracks as needed
#         mTracker._assignDetectionsToTracks()
#
#         # # Clean up old tracks
#         mTracker._deleteLostTracks()
#
#         # print("track centroids\t", end='')
#         # for track in mTracker.tracks:
#         #     print(track.centroid.pt, end='')
#         # print('')
#
#
#
#         keypoints = [track.centroid for track in mTracker.tracks]
#
#         frameWithKeypoints = cv.drawKeypoints(frameWithKeypoints, keypoints, outImage=np.array([]), color=(0, 0, 255),
#                                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#         mTracker.fg = cv.drawKeypoints(mTracker.fg, keypoints, outImage=np.array([]), color=(0, 0, 255),
#                                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         cv.imshow('fg', mTracker.fg)
#         cv.imshow('frame', frameWithKeypoints)
#
#         if cv.waitKey(20) & 0xFF == ord('q'):
#             break
