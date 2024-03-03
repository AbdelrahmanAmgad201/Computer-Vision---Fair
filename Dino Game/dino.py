import cv2
import numpy as np
import mediapipe as mp
import pyautogui
       
class handDetector(): # made a seperate class so we can use it in different applications
    def __init__(self, mode=False, maxHands=1,complexity=1, detectionCon=0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList


wCam, hCam = 640, 480 #dimensions of the camera window can be changed

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = handDetector(detectionCon=0.75) 
tipIds = [4, 8, 12, 16, 20] 

while True:
    success, img = cap.read()
    img = detector.findHands(img) # drawing on the hand
    lmList = detector.findPosition(img, draw=False) # returns the values of X,Y of each ID

    if len(lmList) != 0: 
        count = 0
        
        # Rest of Fingers ( if upper tip is above higher tip increment count by 1)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                count += 1
            
        # Thumb (different technique) ( using X values instead of Y)
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            count += 1
            
        if count >= 4:
            pyautogui.press('space')




    cv2.imshow("Image", img)
    cv2.waitKey(1)