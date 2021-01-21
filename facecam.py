import cv2
import numpy 
import face_recognition
import os
from datetime import *

path = 'images'
img = []
classnames = []
mylist = os.listdir(path)
print(mylist)
for clas in mylist:
    curimg = cv2.imread(f'{path}/{clas}')
    img.append(curimg)
    classnames.append(os.path.splitext(clas)[0])
print(classnames)

def findencodings(img):
    encodelist = []
    for iimg in img:
        img = cv2.cvtColor(iimg,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(iimg)[0]
        encodelist.append(encode)
    return encodelist

def markAttendence(name):
    with open('attendence.csv','r+') as f:
        myDatalist = f.readlines()
        namelist=[]
        for line in myDatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            f.writelines(f'\n{name},{now}')

encodeListKnown = findencodings(img)
print(len(encodeListKnown))

cap = cv2.VideoCapture(0)

while True:
    success,iimg = cap.read()
    imgS = cv2.resize(iimg,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceloc in zip(encodeCurFrame,facesCurFrame) :
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = numpy.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4 
            cv2.rectangle(iimg,(x1,y1),(x2,y2),(0,0,225),2)
            cv2.rectangle(iimg,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(iimg,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
            markAttendence(name)

    cv2.imshow('webcam',iimg)
    cv2.waitKey(1)  