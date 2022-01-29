import cv2


class Utils:

    def __init__(self):
        self.pointsList = []

    def mousePoints(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
            self.pointsList.append([x, y])
            print(self.pointsList)
            #print(x, y)

    def gradient(pt1, pt2):
        return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

    def getAngle():
        print("angle")
        pt1, pt2, pt3 = pointsList[-3:]
        print(pt1, pt2, pt3)
        m1 = gradient(pt1, pt2)
        m2 = gradient(pt1, pt3)
        angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
        angD = round(math.degrees(angR))
        print(angD)
        return angD