from django.shortcuts import redirect, render
from .Detector import HandDetector
import cv2
import mediapipe as mp


import cv2
import numpy as np
from pyzbar.pyzbar import decode


def qrcode(request):
    #img = cv2.imread('1.png')
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        success, img = cap.read()
        for barcode in decode(img):
            myData = barcode.data.decode('utf-8')
            # print(myData)
            pts = np.array([barcode.polygon],np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,255,0),5)
            pts2 = barcode.rect
            cv2.putText(img,myData,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2)

        cv2.imshow('Image',img)
        cv2.waitKey(1)
        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            break

    return redirect('/')






import cv2
import mediapipe as mp
import pyautogui

def eye(request):
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    screen_w, screen_h = pyautogui.size()
    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))
            if (left[0].y - left[1].y) < 0.004:
                pyautogui.click()
                pyautogui.sleep(1)
        cv2.imshow('Image', frame)
        cv2.waitKey(1)
        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
            cam.release()
            cv2.destroyAllWindows()
            break

    return redirect('/')





from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np


def power(request):

    # Parameters
    width, height = 1280, 720
    gestureThreshold = 300
    folderPath = "app/Presentation"

    # Camera Setup
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Hand Detector
    detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

    # Variables
    imgList = []
    delay = 30
    buttonPressed = False
    counter = 0
    drawMode = False
    imgNumber = 0
    delayCounter = 0
    annotations = [[]]
    annotationNumber = -1
    annotationStart = False
    hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

    # Get list of presentation images
    pathImages = sorted(os.listdir(folderPath), key=len)
    print(pathImages)

    while True:
        # Get image frame
        success, img = cap.read()
        img = cv2.flip(img, 1)
        pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
        imgCurrent = cv2.imread(pathFullImage)

        # Find the hand and its landmarks
        hands, img = detectorHand.findHands(img)  # with draw
        # Draw Gesture Threshold line
        cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

        if hands and buttonPressed is False:  # If hand is detected

            hand = hands[0]
            cx, cy = hand["center"]
            lmList = hand["lmList"]  # List of 21 Landmark points
            fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

            # Constrain values for easier drawing
            xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
            indexFinger = xVal, yVal

            if cy <= gestureThreshold:  # If hand is at the height of the face
                if fingers == [1, 0, 0, 0, 0]:
                    print("Left")
                    buttonPressed = True
                    if imgNumber > 0:
                        imgNumber -= 1
                        annotations = [[]]
                        annotationNumber = -1
                        annotationStart = False
                if fingers == [0, 0, 0, 0, 1]:
                    print("Right")
                    buttonPressed = True
                    if imgNumber < len(pathImages) - 1:
                        imgNumber += 1
                        annotations = [[]]
                        annotationNumber = -1
                        annotationStart = False

            if fingers == [0, 1, 1, 0, 0]:
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            if fingers == [0, 1, 0, 0, 0]:
                if annotationStart is False:
                    annotationStart = True
                    annotationNumber += 1
                    annotations.append([])
                print(annotationNumber)
                annotations[annotationNumber].append(indexFinger)
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            else:
                annotationStart = False

            if fingers == [0, 1, 1, 1, 0]:
                if annotations:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True

        else:
            annotationStart = False

        if buttonPressed:
            counter += 1
            if counter > delay:
                counter = 0
                buttonPressed = False

        for i, annotation in enumerate(annotations):
            for j in range(len(annotation)):
                if j != 0:
                    cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

        imgSmall = cv2.resize(img, (ws, hs))
        h, w, _ = imgCurrent.shape
        imgCurrent[0:hs, w - ws: w] = imgSmall

        cv2.imshow("Slides", imgCurrent)
        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            break

    return redirect('/')








def main(request):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(detectionCon=0.5, maxHands=2)
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            centerPoint1 = hand1['center']
            handType1 = hand1["type"]
            fingers1 = detector.fingersUp(hand1)
            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                centerPoint2 = hand2['center']
                handType2 = hand2["type"]
                fingers2 = detector.fingersUp(hand2)
                length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            break

    return redirect('/')



def hand(request):
    return render(request,'x.html')


from .Detector import FaceDetector
import cv2
def face(request):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = FaceDetector()
    
    while True:
        success, img = cap.read()
        img, faces = detector.findFaces(img)
        if faces:
            print(faces[0])
        # if bboxs:
        #     center = bboxs[0]["center"]
        #     cv2.circle(img, center, 5, (82, 167, 54), cv2.FILLED)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            break

    return redirect('/')






from .Detector import FaceMeshDetector

def facemeshdetector(request):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if faces:
            print(faces[0])
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            break

    return redirect('/')







    


import cv2
from .cvzone import *
# import cvzone
from .cvzone.HandTrackingModule import HandDetector as H
from time import sleep
import numpy as np
from pynput.keyboard import Controller


def key(request):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)


    detector = H(detectionCon=1, maxHands=1)
    keyboard_keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
                    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
    final_text = ""
    keyboard = Controller()
    def draw(img, buttonList):
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            cornerRect(img, (button.pos[0], button.pos[1],
                                                    button.size[0],button.size[0]), 20 ,rt=0)
            cv2.rectangle(img, button.pos, (x + w, y + h),
                        (100, 120, 100), cv2.FILLED)
                        # coler        
            cv2.putText(img, button.text, (x + 20, y + 65),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
        return img
    def transparent_layout(img, buttonList):
        imgNew = np.zeros_like(img, np.uint8)
        for button in buttonList:
            x, y = button.pos
            cornerRect(imgNew, (button.pos[0], button.pos[1],
                                                    button.size[0],button.size[0]), 20 ,rt=0)
            cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),
                                    (255, 144, 30), cv2.FILLED)
            cv2.putText(imgNew, button.text, (x + 20, y + 65),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
        out = img.copy()
        alpaha = 0.5
        mask = imgNew.astype(bool)
        print(mask.shape)
        out[mask] = cv2.addWeighted(img, alpaha, imgNew, 1-alpaha, 0)[mask]
        return out


    class Button():
        def __init__(self, pos, text, size=[85, 85]):
            self.pos = pos
            self.size = size
            self.text = text

    buttonList = []
    # mybutton = Button([100, 100], "Q")
    for k in range(len(keyboard_keys)):
        for x, key in enumerate(keyboard_keys[k]):
            buttonList.append(Button([100 * x + 25, 100 * k + 50], key))


    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bboxInfo = detector.findPosition(img)
        img = draw(img, buttonList)  # change the draw funtion to transparent_layout for transparent keys
        if lmList:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size
                if x < lmList[8][0]<x+w and y < lmList[8][1] < y+h:
                    cv2.rectangle(img, button.pos, (x + w, y + h),
                                (0, 255, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                    l, _, _ = detector.findDistance(8,12, img, draw=False)
                    print(l)
                    if l < 25:
                        keyboard.press(button.text)
                        cv2.rectangle(img, button.pos, (x + w, y + h),
                                    (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                        final_text += button.text
                        sleep(0.20)
        cv2.rectangle(img, (25,350), (700, 450),
                    (255, 255, 255), cv2.FILLED)
        cv2.putText(img, final_text, (60, 425),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
        cv2.imshow("output", img)
        cv2.waitKey(1)

        if cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            break

    return redirect('/')








# import math
# import random
# from cvzone.HandTrackingModule import HandDetector as H
# import Detector
# HandDetector = Detector.HandDetector



# from .Detector import HandDetector as b
# def sac(request):


#     class SnakeGameClass:
#         def __init__(self) -> None:
#             pass
#         def __init__(self, pathFood):
#             self.points = []  # all points of the snake
#             self.lengths = []  # distance between each point
#             self.currentLength = 0  # total length of the snake
#             self.allowedLength = 150  # total allowed Length
#             self.previousHead = 0, 0  # previous head point

#             self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
#             self.hFood, self.wFood, _ = self.imgFood.shape
#             self.foodPoint = 0, 0
#             self.randomFoodLocation()

#             self.score = 0
#             self.gameOver = False

#         def randomFoodLocation(self):
#             self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

#         def update(self, imgMain, currentHead):

#             if self.gameOver:
#                 cvzone.putTextRect(imgMain, "Game Over", [300, 400],
#                                 scale=7, thickness=5, offset=20)
#                 cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [300, 550],
#                                 scale=7, thickness=5, offset=20)
#             else:
#                 px, py = self.previousHead
#                 cx, cy = currentHead

#                 self.points.append([cx, cy])
#                 distance = math.hypot(cx - px, cy - py)
#                 self.lengths.append(distance)
#                 self.currentLength += distance
#                 self.previousHead = cx, cy

#                 # Length Reduction
#                 if self.currentLength > self.allowedLength:
#                     for i, length in enumerate(self.lengths):
#                         self.currentLength -= length
#                         self.lengths.pop(i)
#                         self.points.pop(i)
#                         if self.currentLength < self.allowedLength:
#                             break

#                 # Check if snake ate the Food
#                 rx, ry = self.foodPoint
#                 if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
#                         ry - self.hFood // 2 < cy < ry + self.hFood // 2:
#                     self.randomFoodLocation()
#                     self.allowedLength += 50
#                     self.score += 1
#                     print(self.score)

#                 # Draw Snake
#                 if self.points:
#                     for i, point in enumerate(self.points):
#                         if i != 0:
#                             cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
#                     cv2.circle(imgMain, self.points[-1], 20, (0, 255, 0), cv2.FILLED)

#                 # Draw Food
#                 imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
#                                             (rx - self.wFood // 2, ry - self.hFood // 2))

#                 cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80],
#                                 scale=3, thickness=3, offset=10)

#                 # Check for Collision
#                 pts = np.array(self.points[:-2], np.int32)
#                 pts = pts.reshape((-1, 1, 2))
#                 cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
#                 minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

#                 if -1 <= minDist <= 1:
#                     print("Hit")
#                     self.gameOver = True
#                     self.points = []  # all points of the snake
#                     self.lengths = []  # distance between each point
#                     self.currentLength = 0  # total length of the snake
#                     self.allowedLength = 150  # total allowed Length
#                     self.previousHead = 0, 0  # previous head point
#                     self.randomFoodLocation()

#             return imgMain




#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     # cap.set(3, 1280)
#     # cap.set(4, 720)

#     detector = b(detectionCon=0.8)


#     game = SnakeGameClass("app/Donut2.png")

#     while True:
#         success, img = cap.read()
#         img = cv2.flip(img, 1)
#         hands, img = detector.findHands(img, flipType=False)

#         if hands:
#             lmList = hands[0]['lmList']
#             pointIndex = lmList[8][0:2]
#             img = game.update(img, pointIndex)
#         cv2.imshow("Image", img)
#         key = cv2.waitKey(1)
#         if key == ord('r'):
#             game.gameOver = False