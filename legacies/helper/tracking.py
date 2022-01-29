import cv2 as cv

pointsList = []

class tracking:

    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    # function for detecting left mouse click
    def click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print("Pressed", x, y)
            point = (x, y)