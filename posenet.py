import cv2
import mediapipe as mp
import numpy as np
from GraphConvolution import relativeMiddleCor
def media_posenet(image1):
    mp_drawing = mp.solutions.drawing_utils

    # 参数：1、颜色，2、线条粗细，3、点的半径
    DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 2, 2)
    DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 2, 2)
    mypose=mp.solutions.pose

    pose=mypose.Pose()
    # mp.solutions.holistic是一个类别，是人的整体
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(static_image_mode=True)
    image=np.asarray(image1)

    image_hight, image_width, _ = image.shape
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    #if results.pose_landmarks:
    #     print(
    #         f'Nose coordinates: ('
    #         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
    #         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
    #     )
    #
    # print(results.pose_landmarks)
    a=[]
    for id ,lm in enumerate(results.pose_landmarks.landmark):
        ih, iw, ic = image.shape
        x, y = int(lm.x * iw), int(lm.y * ih)
        w=(id, x, y)
        a.append(w)
    #a=np.array(a)
    return a
def openpose_net(image):
    global sum_points
    protoFile = "pose_deploy_linevec.prototxt"
    weightsfile = "pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsfile)

    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inWidth = im.shape[1]
    inHeight = im.shape[0]
    netInputSize = (368, 368)
    inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    scaleX = float(inWidth) / output.shape[3]
    scaleY = float(inHeight) / output.shape[2]
    points_x = []
    points_y=[]
    # Confidence treshold
    threshold = 0.1
    pose_array = np.zeros([2, 18])

    for i in range(nPoints):
        # Obtain probability map
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = scaleX * point[0]
        y = scaleY * point[1]

        if prob > threshold:
            # Add the point to the list if the probability is greater than the threshold

            pose_array[:, i] = int(x), int(y)
            points_x.append(int(x))
            points_y.append(int(y))

        else:
            points_x.append(0)
            points_y.append(0)
            #pose_array[:, i] = 0,0

        sum_points=relativeMiddleCor(points_x,points_y)

    return sum_points

