#17/7/2022
#Bandar

import cv2
import easyocr
import numpy as np

def processData(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blackLines = []
    blackLineThreshold = 5

    #Y trimming
    for i in range(len(frame)):
        blackLineCurrent = 0
        for j in range(len(frame[i])):
            if frame[i][j] < 75: #75 = dark gray
                blackLineCurrent += 1 

        if blackLineCurrent > blackLineThreshold:
            blackLines.append(i)
            #frame = cv2.line(frame, (0, i), (len(frame[i]), i), (0, 0, 255), thickness=1)

    newFrame = []
    for i in range(len(blackLines)):
        newFrame.append(frame[blackLines[i]])

    frame = newFrame

    #X trimming
    blackLines = []
    for i in range(len(frame[0])):
        blackLineCurrent = 0
        for j in range(len(frame)):
            if frame[j][i] < 75:
                blackLineCurrent += 1 

        if blackLineCurrent > blackLineThreshold:
            blackLines.append(i)
            #frame = cv2.line(frame, (i, 0), (i, len(frame[0])), (0, 0, 255), thickness=1)
    
    #cv2.imwrite('userTakenImage.png', frame)
    
    newFrame = []
    for i in range(len(frame)):
        newFrame.append(frame[i][min(blackLines):max(blackLines)])
    
    newFrame = np.array(newFrame)
    newFrame = cv2.resize(newFrame, (500, 700)) 
    
    edged = cv2.Canny(newFrame, 30, 200)
    
    contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 800 and w > 350:
            cv2.rectangle(newFrame, (x, y), (x+w, y+h), (0,0,0), 2)

    cv2.imwrite('userTakenImage.png', newFrame)

    return newFrame
    

def readImage(frame, reader):
    data = reader.readtext(frame, paragraph="False")
    text = []
    for i in range(len(data)):
        l = data[i][-1].split()
        for j in range(len(l)):
            text.append(l[j])
            frame = cv2.rectangle(frame, data[i][0][0], data[i][0][2], (0,0,255), 1) #Colour is in BGR not RGB 
            #cv2.imwrite('userTakenImage.png', frame)
    
    return text

def runCamera():
    cap = cv2.VideoCapture(0) 
    while(True): 
        _, frame = cap.read() #_ = throwaway variable (just returns true if frame is available)

        cv2.imshow("preview", frame)
        if cv2.waitKey(1) == 113: #ASCII for "q" 
            break
        elif cv2.waitKey(1) == 32: #ASCII for space
            reader = easyocr.Reader(['en'])
            frame = cv2.imread("IMG_8728.jpg")
            
            frame = processData(frame)
            result = readImage(frame, reader)
            print(result)

            cv2.destroyAllWindows()

    cv2.destroyAllWindows()

runCamera()

