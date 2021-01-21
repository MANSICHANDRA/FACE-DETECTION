import cv2
import numpy 
import face_recognition
import sys

imgssr = face_recognition.load_image_file("images/shushant.jpg")
imgssr = cv2.cvtColor(imgssr,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('images/test.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgssr)[0]
encodessr = face_recognition.face_encodings(imgssr)[0]
cv2.rectangle(imgssr,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloc = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodessr],encodetest)
facedis = face_recognition.face_distance([encodessr],encodetest)   
print(results,facedis)
cv2.putText(imgtest,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('shushant',imgssr)
cv2.imshow('testa',imgtest)
cv2.waitKey(0)
sys.exit(1)