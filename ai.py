import cv2
import imutils
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import cvzone
cap = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.2)

keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"]]
        
buttonList = []  # Create an empty list to store the button information
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append([100 * j + 50, 100 * i + 50, 100, 100, key])  # Store button coordinates and text

finalText = ""  # Variable to store the final text input
prevFingerPos = None

while True:
    success, img = cap.read()
    img = imutils.resize(img, width=1280, height=720)
    hands, img = detector.findHands(img)  # Find the hands and their landmarks

    # Draw the buttons on the image
    for button in buttonList:
        x, y, w, h, text = button
        cv2.rectangle(img, (x, y), (x + w, y + h), (175, 0, 175), cv2.FILLED)
        cv2.putText(img, text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    if hands:
        for hand in hands:
            lmList = hand["lmList"]
            bboxInfo = hand["bbox"]
            fingerPos = lmList[8]

            for button in buttonList:
                x, y, w, h, text = button

                if x < fingerPos[0] < x + w and y < fingerPos[1] < y + h:
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    l, _, _ = detector.findDistance(8, 12, img, draw=False)
                    print(l)
                    # When Clicked
                    if l < 70:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        finalText += text
                        sleep(0.2)

            if prevFingerPos is not None and lmList[8][1] < prevFingerPos[1]:  # If the index finger is up (going down to up)
                cv2.line(img, prevFingerPos[:2], lmList[8][:2], (0, 0, 0), 10)

            prevFingerPos = fingerPos

    cv2.rectangle(img, (50, 350), (1000, 450), (175, 10, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 425),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
