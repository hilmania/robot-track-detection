import cv2 as cv
from math import atan2, cos, sin, sqrt, pi, atan, degrees
import numpy as np

pointsList = []

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
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
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]

    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

    return angle

def mousePoints (event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), 5, (0, 0, 255), cv.FILLED)
        pointsList.append([x, y])
        #print(pointsList)
        #print(x, y)
    return pointsList

def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

def getAngle(pointsList):
    #print("angle")
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(degrees(angR))
    print(angD)
    return angD


img = cv.imread("input_img.jpg")

# Was the image there?
if img is None:
    print("Error: File not found")
    exit(0)

#cv.imshow('Input Image', img)

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Convert image to binary
_, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# Find all the contours in the thresholded image
contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for i, c in enumerate(contours):

    # Calculate the area of each contour
    area = cv.contourArea(c)

    # Ignore contours that are too small or too large
    if area < 3700 or 100000 < area:
        continue

    # Draw each contour only for visualisation purposes
    cv.drawContours(img, contours, i, (0, 0, 255), 2)

    # Calculate orientation between 0 and 180
    # cv.minAreaRect returns:
    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
    # rect = cv.minAreaRect(c)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    #
    # # Retrieve the key parameters of the rotated bounding box
    # center = (int(rect[0][0]), int(rect[0][1]))
    # width = int(rect[1][0])
    # height = int(rect[1][1])
    # angle = int(rect[2])
    #
    # if width < height:
    #     angle = 90 - angle
    # else:
    #     angle = -angle
    #
    # label = "  Rotation Angle: " + str(angle) + " degrees"
    # textbox = cv.rectangle(img, (center[0] - 35, center[1] - 25),
    #                        (center[0] + 295, center[1] + 10), (255, 255, 255), -1)
    # cv.putText(img, label, (center[0] - 50, center[1]),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)
    # cv.drawContours(img, [box], 0, (0, 0, 255), 2)

    # Find the orientation of each shape
    getOrientation(c, img)

while True:
    cv.imshow('Output Image', img)
    cv.setMouseCallback('Output Image', mousePoints)

    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)

    key = cv.waitKey(1) & 0xFF
    if key == ord("r"):
        pointsList = []
        #img = cv.imread("input_img.jpg")
    elif key == ord("q"):
        break

cv.destroyAllWindows()

# Save the output image to the current directory
cv.imwrite("output_img.jpg", img)