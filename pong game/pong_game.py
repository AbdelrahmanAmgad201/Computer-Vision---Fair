import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
# set width and height
cap.set(3, 1280)   # this is width
cap.set(4, 720)    # this is height

imgBackground = cv2.imread("/home/abdelrahman-amgad/dev/MindCloud/Computer-Vision---Fair/pong game/Resources/Background.png")
imgGameOver = cv2.imread("/home/abdelrahman-amgad/dev/MindCloud/Computer-Vision---Fair/pong game/Resources/gameOver.png")
imgBall = cv2.imread("/home/abdelrahman-amgad/dev/MindCloud/Computer-Vision---Fair/pong game/Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("/home/abdelrahman-amgad/dev/MindCloud/Computer-Vision---Fair/pong game/Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("/home/abdelrahman-amgad/dev/MindCloud/Computer-Vision---Fair/pong game/Resources/bat2.png", cv2.IMREAD_UNCHANGED)

# hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# global variables definition

ballPos = [100, 100]
speedX = 25
speedY = 25
gameOver = False
score = [0, 0]



while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False )  # flipType = false to flip the hand position

    # overlaying the background image
    img = cv2.addWeighted(img, 0.25, imgBackground, 0.75, 0)   #control with transparency and when adding the two values =1

    # check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape   
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)  # determine min and max hight for bats can move

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1

    # when game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        if ballPos[0] < 40:  # Player 2 wins
            winner_text = "Player 2 Wins!"
            score_display = str(score[1]).zfill(2)
        else:  # Player 1 wins
            winner_text = "Player 1 Wins!"
            score_display = str(score[0]).zfill(2)

        cv2.putText(img, winner_text, (400, 75), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(img, score_display, (585, 360), cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)


    # if game not over move the ball
    else:

        # move the Ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)

    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 25
        speedY = 25
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread("/home/abdelrahman-amgad/dev/MindCloud/Computer-Vision---Fair/pong game/Resources/gameOver.png")
    elif key == ord('q'):
        break

# Destroy all OpenCV windows before exiting
cv2.destroyAllWindows()
