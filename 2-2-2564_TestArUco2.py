#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import cv2.aruco as aruco
device = 0

# read the image
#img = cv2.imread('triangle.jpg')
cap = cv2.VideoCapture(device)
cap.set(3, 720)
cap.set(4, 1280)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

firstMarkerID = None
secondMarkerID = None

matrix_coefficients = [[1.25019753e+03, 0.00000000e+00, 8.31027626e+02,], 
                       [0.00000000e+00, 1.24629197e+03, 4.93761478e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
distortion_coefficients = [[ 0.22126322, -0.38411381, -0.01361529,  0.03204828, -0.06211731]]


def track(matrix_coefficients, distortion_coefficients):
    pointCircle = (0, 0)
    markerTvecList = []
    markerRvecList = []
    composedRvec, composedTvec = None, None
    while True:
        ret, frame = cap.read()
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters

        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                parameters=parameters,
                                                                cameraMatrix=np.float32(matrix_coefficients),
                                                                distCoeff=np.float32(distortion_coefficients))
        
        if np.all(ids is not None):  # If there are markers found by detector
            del markerTvecList[:]
            del markerRvecList[:]
            zipped = zip(ids, corners)
            ids, corners = zip(*(sorted(zipped)))
            axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, np.float32(matrix_coefficients),
                                                                           np.float32(distortion_coefficients))

                if ids[i] == firstMarkerID:
                    firstRvec = rvec
                    firstTvec = tvec
                    isFirstMarkerCalibrated = True
                    firstMarkerCorners = corners[i]
                elif ids[i] == secondMarkerID:
                    secondRvec = rvec
                    secondTvec = tvec
                    isSecondMarkerCalibrated = True
                    secondMarkerCorners = corners[i]

                # print(markerPoints)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                markerRvecList.append(rvec)
                markerTvecList.append(tvec)

                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                
           

            if len(ids) > 1 and composedRvec is not None and composedTvec is not None:
                info = cv2.composeRT(composedRvec, composedTvec, secondRvec.T, secondTvec.T)
                TcomposedRvec, TcomposedTvec = info[0], info[1]

                objectPositions = np.array([(0, 0, 0)], dtype=np.float)  # 3D point for projection
                imgpts, jac = cv2.projectPoints(axis, TcomposedRvec, TcomposedTvec, np.float32(matrix_coefficients),
                                                np.float32(distortion_coefficients))

                # frame = draw(frame, corners[0], imgpts)
                aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, TcomposedRvec, TcomposedTvec,
                               0.01)  # Draw Axis
                relativePoint = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
                cv2.circle(frame, relativePoint, 2, (255, 255, 0))


        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Calibration
            if len(ids) > 1:  # If there are two markers, reverse the second and get the difference
                firstRvec, firstTvec = firstRvec.reshape((3, 1)), firstTvec.reshape((3, 1))
                secondRvec, secondTvec = secondRvec.reshape((3, 1)), secondTvec.reshape((3, 1))

                composedRvec, composedTvec = relativePosition(firstRvec, firstTvec, secondRvec, secondTvec)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

track(matrix_coefficients, distortion_coefficients)


# In[ ]:




