import numpy as np
import cv2
import serial
import time
import math
from math import atan2, cos, sin, sqrt, pi, atan, degrees
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Create the circle
colour = (0, 0, 255)
# colour = (0, 255, 81)
lineWidth = -1  # -1 will result in filled circle
radius = 15
point = (0, 0)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=7)

trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0

pointsList = [(0,0)] * 3
global_angle = 0
CM_TO_PIXEL = 32.0 / 640

arduino = serial.Serial(port='COM10', baudrate=115200, timeout=.1)

# function for detecting left mouse click
def point_click(event, x, y, flags, param):
    global point, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pressed", x, y)
        point = (x, y)
        pointsList[1] = point

# function for calculate distance
def distance(x1, y1, x2, y2):
    """
    Calculate distance between two points
    """
    dist = math.sqrt(math.fabs(x2 - x1) ** 2 + math.fabs(y2 - y1) ** 2)
    return round(dist * CM_TO_PIXEL, 2)

# function for finding color match
def find_color1(frame):
    """
    Filter "frame" for HSV bounds for color1 (inplace, modifies frame) & return coordinates of the object with that color
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_lowerbound = np.array([139, 149, 131])  # replace THIS LINE w/ your hsv lowerb
    hsv_upperbound = np.array([179, 255, 219])  # replace THIS LINE w/ your hsv upperb
    mask = cv2.inRange(hsv_frame, hsv_lowerbound, hsv_upperbound)
    res = cv2.bitwise_and(frame, frame, mask=mask)  # filter inplace
    cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        maxcontour = max(cnts, key=cv2.contourArea)

        # Find center of the contour
        M = cv2.moments(maxcontour)
        if M['m00'] > 0 and cv2.contourArea(maxcontour) > 1000:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy), True
        else:
            return (700, 700), False  # faraway point
    else:
        return (700, 700), False  # faraway point

# function for finding color match
def find_color2(frame):
    """
    Filter "frame" for HSV bounds for color1 (inplace, modifies frame) & return coordinates of the object with that color
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_lowerbound = np.array([101, 152, 92])  # replace THIS LINE w/ your hsv lowerb
    hsv_upperbound = np.array([149, 255, 243])  # replace THIS LINE w/ your hsv upperb
    mask = cv2.inRange(hsv_frame, hsv_lowerbound, hsv_upperbound)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        maxcontour = max(cnts, key=cv2.contourArea)

        # Find center of the contour
        M = cv2.moments(maxcontour)
        if M['m00'] > 0 and cv2.contourArea(maxcontour) > 2000:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy), True  # True
        else:
            return (700, 700), True  # faraway point
    else:
        return (700, 700), True  # faraway point


def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def get_angle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(degrees(angR))
    #print(angD)
    cv2.putText(img, str(angD), (pt1[0] - 40, pt1[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
    return angD

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    ## [visualization1]

def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    #print(p2)

    drawAxis(img, cntr, p1, (255, 255, 0), 9)
    # drawAxis(img, cntr, pointsList[1], (0, 0, 255), 5)
    cv2.line(img, cntr, pointsList[1], (0, 0, 255), 3, cv2.LINE_AA)




    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]

    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv2.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Angle : " + str(-int(np.rad2deg(angle)) - 90) + " degree", (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2 )

    global_angle = angle
    return angle

def fuzzy_robot(d, a):
    distance = ctrl.Antecedent(np.arange(0, 150, 1), 'distance')
    uni_angle = np.array([-180, -90, -30, 0, 30, 90, 180])
    angle = ctrl.Antecedent(uni_angle, 'angle')

    uni_velocity = np.array([0, 55, 105, 155, 205, 255])
    velocity = ctrl.Consequent(uni_velocity, 'velocity')

    distance['sangat_dekat'] = fuzz.trapmf(distance.universe, [0, 0, 25, 50])
    distance['dekat'] = fuzz.trimf(distance.universe, [25, 50, 75])
    distance['cukup'] = fuzz.trimf(distance.universe, [50, 75, 100])
    distance['jauh'] = fuzz.trimf(distance.universe, [75, 100, 125])
    distance['sangat_jauh'] = fuzz.trapmf(distance.universe, [100, 125, 150, 150])

    # distance.view()
    # plt.title("Distance Membership Function")

    angle['sangat_kiri'] = fuzz.trapmf(angle.universe, [-180, -180, -90, -30])
    angle['kiri'] = fuzz.trimf(angle.universe, [-90, -30, 0])
    angle['lurus'] = fuzz.trimf(angle.universe, [-30, 0, 30])
    angle['kanan'] = fuzz.trimf(angle.universe, [0, 30, 90])
    angle['sangat_kanan'] = fuzz.trapmf(angle.universe, [30, 90, 180, 180])

    # angle.view()
    # plt.title("Angle Membership Function")

    velocity['stop'] = fuzz.trapmf(velocity.universe, [0, 0, 55, 105])
    velocity['very_slow'] = fuzz.trimf(velocity.universe, [55, 105, 155])
    velocity['slow'] = fuzz.trimf(velocity.universe, [105, 155, 205])
    velocity['normal'] = fuzz.trimf(velocity.universe, [155, 205, 255]) 
    velocity['fast'] = fuzz.trimf(velocity.universe, [205, 255, 255])

    # velocity.view()
    # plt.title("Velocity Membership Function")

    # rule_roda_kanan
    rule1_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['sangat_kiri'], velocity['stop'])
    rule2_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['kiri'], velocity['very_slow'])
    rule3_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['lurus'], velocity['very_slow'])
    rule4_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['kanan'], velocity['slow'])
    rule5_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['sangat_kanan'], velocity['very_slow'])

    rule6_roda_kanan = ctrl.Rule(distance['dekat'] | angle['sangat_kiri'], velocity['very_slow'])
    rule7_roda_kanan = ctrl.Rule(distance['dekat'] | angle['kiri'], velocity['very_slow'])
    rule8_roda_kanan = ctrl.Rule(distance['dekat'] | angle['lurus'], velocity['slow'])
    rule9_roda_kanan = ctrl.Rule(distance['dekat'] | angle['kanan'], velocity['slow'])
    rule10_roda_kanan = ctrl.Rule(distance['dekat'] | angle['sangat_kanan'], velocity['slow'])

    rule11_roda_kanan = ctrl.Rule(distance['cukup'] | angle['sangat_kiri'], velocity['slow'])
    rule12_roda_kanan = ctrl.Rule(distance['cukup'] | angle['kiri'], velocity['slow'])
    rule13_roda_kanan = ctrl.Rule(distance['cukup'] | angle['lurus'], velocity['normal'])
    rule14_roda_kanan = ctrl.Rule(distance['cukup'] | angle['kanan'], velocity['normal'])
    rule15_roda_kanan = ctrl.Rule(distance['cukup'] | angle['sangat_kanan'], velocity['normal'])

    rule16_roda_kanan = ctrl.Rule(distance['jauh'] | angle['sangat_kiri'], velocity['slow'])
    rule17_roda_kanan = ctrl.Rule(distance['jauh'] | angle['kiri'], velocity['normal'])
    rule18_roda_kanan = ctrl.Rule(distance['jauh'] | angle['lurus'], velocity['normal'])
    rule19_roda_kanan = ctrl.Rule(distance['jauh'] | angle['kanan'], velocity['normal'])
    rule20_roda_kanan = ctrl.Rule(distance['jauh'] | angle['sangat_kanan'], velocity['fast'])

    rule21_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['sangat_kiri'], velocity['normal'])
    rule22_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['kiri'], velocity['normal'])
    rule23_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['lurus'], velocity['normal'])
    rule24_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['kanan'], velocity['fast'])
    rule25_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['sangat_kanan'], velocity['fast'])

    # rule roda kiri
    rule1_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['sangat_kiri'], velocity['very_slow'])
    rule2_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['kiri'], velocity['slow'])
    rule3_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['lurus'], velocity['slow'])
    rule4_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['kanan'], velocity['very_slow'])
    rule5_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['sangat_kanan'], velocity['stop'])

    rule6_roda_kiri = ctrl.Rule(distance['dekat'] | angle['sangat_kiri'], velocity['slow'])
    rule7_roda_kiri = ctrl.Rule(distance['dekat'] | angle['kiri'], velocity['slow'])
    rule8_roda_kiri = ctrl.Rule(distance['dekat'] | angle['lurus'], velocity['normal'])
    rule9_roda_kiri = ctrl.Rule(distance['dekat'] | angle['kanan'], velocity['very_slow'])
    rule10_roda_kiri = ctrl.Rule(distance['dekat'] | angle['sangat_kanan'], velocity['very_slow'])

    rule11_roda_kiri = ctrl.Rule(distance['cukup'] | angle['sangat_kiri'], velocity['slow'])
    rule12_roda_kiri = ctrl.Rule(distance['cukup'] | angle['kiri'], velocity['normal'])
    rule13_roda_kiri = ctrl.Rule(distance['cukup'] | angle['lurus'], velocity['normal'])
    rule14_roda_kiri = ctrl.Rule(distance['cukup'] | angle['kanan'], velocity['slow'])
    rule15_roda_kiri = ctrl.Rule(distance['cukup'] | angle['sangat_kanan'], velocity['slow'])

    rule16_roda_kiri = ctrl.Rule(distance['jauh'] | angle['sangat_kiri'], velocity['slow'])
    rule17_roda_kiri = ctrl.Rule(distance['jauh'] | angle['kiri'], velocity['normal'])
    rule18_roda_kiri = ctrl.Rule(distance['jauh'] | angle['lurus'], velocity['slow'])
    rule19_roda_kiri = ctrl.Rule(distance['jauh'] | angle['kanan'], velocity['slow'])
    rule20_roda_kiri = ctrl.Rule(distance['jauh'] | angle['sangat_kanan'], velocity['normal'])

    rule21_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['sangat_kiri'], velocity['normal'])
    rule22_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['kiri'], velocity['normal'])
    rule23_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['lurus'], velocity['slow'])
    rule24_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['kanan'], velocity['normal'])
    rule25_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['sangat_kanan'], velocity['fast'])

    # rule_roda_kiri
    robot_ctrl_right = ctrl.ControlSystem(
        [rule1_roda_kanan, rule2_roda_kanan, rule3_roda_kanan, rule4_roda_kanan, rule5_roda_kanan,
        rule6_roda_kanan, rule7_roda_kanan, rule8_roda_kanan, rule9_roda_kanan, rule10_roda_kanan,
        rule11_roda_kanan, rule12_roda_kanan, rule13_roda_kanan, rule14_roda_kanan, rule15_roda_kanan,
        rule16_roda_kanan, rule17_roda_kanan, rule18_roda_kanan, rule19_roda_kanan, rule20_roda_kanan,
        rule21_roda_kanan, rule22_roda_kanan, rule23_roda_kanan, rule24_roda_kanan, rule25_roda_kanan]
    )

    robot_ctrl_left = ctrl.ControlSystem(
        [rule1_roda_kiri, rule2_roda_kiri, rule3_roda_kiri, rule4_roda_kiri, rule5_roda_kiri,
        rule6_roda_kiri, rule7_roda_kiri, rule8_roda_kiri, rule9_roda_kiri, rule10_roda_kiri,
        rule11_roda_kiri, rule12_roda_kiri, rule13_roda_kiri, rule14_roda_kiri, rule15_roda_kiri,
        rule16_roda_kiri, rule17_roda_kiri, rule18_roda_kiri, rule19_roda_kiri, rule20_roda_kiri,
        rule21_roda_kiri, rule22_roda_kiri, rule23_roda_kiri, rule24_roda_kiri, rule25_roda_kiri]
    )

    wheel_right = ctrl.ControlSystemSimulation(robot_ctrl_right)
    wheel_left = ctrl.ControlSystemSimulation(robot_ctrl_left)

    wheel_right.input['distance'] = d
    wheel_right.input['angle'] = a

    wheel_left.input['distance'] = d
    wheel_left.input['angle'] = a

    wheel_right.compute()
    wheel_left.compute()

    # print("velocity_right_wheel : ", wheel_right.output['velocity'])
    # print("velocity_left_wheel : ", wheel_left.output['velocity'])
    # velocity.view(sim=wheel_right)
    # plt.title("Velocity Right Wheel")
    # velocity.view(sim=wheel_left)
    # plt.title("Velocity Left Wheel")

    # plt.show()
    vel = []
    vel.append(wheel_right.output['velocity'])
    vel.append(wheel_left.output['velocity'])
    return vel

def write_read(x):
    arduino.write(bytes([int(x)]))
    time.sleep(0.05)
    data = arduino.readline()
    return data

# ======================================================================================================================
# main section
# event handler
cv2.namedWindow("Optical Flow")  # must match the imshow 1st argument
cv2.setMouseCallback("Optical Flow", point_click)

cap = cv2.VideoCapture("videos/dataset-1.mp4")
# cap = cv2.VideoCapture(0)

while True:
    # start time to calculate FPS
    start = time.time()

    suc, frame = cap.read()

    cv2.circle(frame, point, radius, colour, lineWidth, cv2.FILLED)  # circle properties as arguments
    frame = cv2.resize(frame, (0, 0), fx=0.99, fy=0.99)  # this fx,fy value will be explained in post

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv2.contourArea(c)

        # Ignore contours that are too small or too large
        if area < 3700 or 100000 < area:
            continue

        # Draw each contour only for visualisation purposes
        cv2.drawContours(frame, contours, i, (0, 0, 255), 2)
        getOrientation(c, frame)


    img = frame.copy()


    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Newest detected point
            # cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), 3)
        trajectories = new_trajectories

        # Draw all the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # Draw coordinate
        cv2.putText(img, "Coordinate : " + str(trajectories[0][0]), (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2 )
        # print(trajectories[0][0][1])
        # print(pointsList[1])
        # if pointsList is not None:
        cv2.putText(img, "Distance : " + str( distance(pointsList[1][0], pointsList[1][1], trajectories[0][0][0], trajectories[0][0][1]) ) + " cm", (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # drawAxis(img, pointsList[0], pointsList[1], (255, 255, 0), 1)

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Latest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 3, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])
        # print(p)
        pointsList[0] = p
    frame_idx += 1
    prev_gray = frame_gray

    # print(pointsList)
    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)

    velo = fuzzy_robot(distance(pointsList[1][0], pointsList[1][1], trajectories[0][0][0], trajectories[0][0][1]), global_angle )
    # print(velo)

    VR = write_read(str(int(round(velo[0],0))))
    VL = write_read(str(int(round(velo[1],0))))
    print(VR)
    print(VL)

    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "VR : " + str(round(velo[0], 2)) + " PWM", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "VL : " + str(round(velo[1], 2)) + " PWM", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()