import imutils
import cv2

class BasicMotionDetector:
    def __init__(self, accumWeight=0.5, deltaTresh=5, minArea=5000):
        self.isv2 = imutils.is_cv2()
        self.accumWeight = accumWeight
        self.deltaTresh = deltaTresh
        self.minArea = minArea

        self.avg = None

    def update(self, image):
        locs = []

        if self.avg is None:
            self.avg = image.astype("float")
            return locs

        cv2.accumulateWeighted(image, self.avg, self.accumWeight)
        frameDelta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))

        thresh = cv2.threshold(frameDelta, self.deltaTresh, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) > self.minArea:
                locs.append(c)

        return locs

    