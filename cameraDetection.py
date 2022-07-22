#17/7/2022
#Bandar

import cv2
import easyocr

def processData(data, frame):
    text = []
    for i in range(len(data)):
        l = data[i][-1].split()
        for j in range(len(l)):
            text.append(l[j])
            frame = cv2.rectangle(frame, data[i][0][0], data[i][0][2], (0,0,255), 1) #Colour is in BGR not RGB 
            cv2.imwrite('userTakenImage.png', frame)
    
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
            frame = cv2.imread("IMG_8728.png")
            
            result = reader.readtext(frame, paragraph="False")
            print(processData(result, frame))
            print(result)
            cv2.destroyAllWindows()

    cv2.destroyAllWindows()

