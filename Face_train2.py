import os
import cv2 as cv
import numpy as np
from mtcnn import MTCNN


#people=['Beyonce Knowles', 'Angelina Jolie', 'Arnold Schwarzenegger', 'Alexandra Daddario']
people=['Arnold Schwarzenegger', 'Alexandra Daddario']


DIR1=r'/home/akhil-joseph/OpenCV/Face_Recognition/Images'
DIR2=r'/home/akhil-joseph/OpenCV/Face_Recognition/Train2'


detector=MTCNN()


def create_train():
    for person in people:
        path=os.path.join(DIR1,person)
        label=people.index(person)
        i=0
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array=cv.imread(img_path)

            faces_rect=detector.detect_faces(img_array)
            if faces_rect==None:
                print('jerry')
                continue
            for results in faces_rect:
                x, y, w, h = results['box']
                faces_roi= img_array[y:y+h,x:x+w]
                faces_roi=cv.resize(faces_roi,(200,200))
                newpath =os.path.join(DIR2,person)
                newpath=os.path.join(newpath,person)
                cv.imwrite(newpath+str(i)+".jpg",faces_roi)
                i=i+1


create_train()
