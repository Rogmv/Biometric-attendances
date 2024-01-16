from keras.models import load_model
import tensorflow as tf
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
import os



#classifier = load_model(r'/home/akhil-joseph/OpenCV/cv/face_recognition.h5')
classifier = load_model(r'face_recognition.h5')

validation_datagen=ImageDataGenerator(rescale=1./255)

os.system('clear')

img_rows, img_cols=224,224
validation_data_dir=r'test'
#validation_data_dir=r'/home/akhil-joseph/OpenCV/cv/test'
batch_size=1

validation_generator= validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows,img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


score=Model.evaluate(classifier,validation_generator,steps=50)

print("The accuracy according to the validaton data :",score[1]*100)

import cv2
import numpy as np

people_class_dict = {
    "[0]": 'Angelina',
    "[1]": 'Arnold '
    }
while True:
    while True:
        b=input("Give a number between 1 and 20 to test the system (-1 to exit) :")
        if int(b)<=20 or int(b)>0:
            break
    if(int(b)==-1):
        break
    path=f'Validate/{b}.jpg'
    print(path)
    input_im=cv2.imread(path)
    cv2.imshow(path,input_im)
    input_original=input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    input_im=cv2.resize(input_im,(224,224),interpolation=cv2.INTER_LINEAR)


    input_im=input_im/255.
    input_im=tf.expand_dims(input_im,axis=1)
    input_im=tf.reshape(input_im,[-1,224,224,3])


    # Get Prediction
    """ res=classifier.predict(input_im, 1, verbose = 0)
    print(res) """
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)

    print("The Model predicts the person to be: ",people_class_dict[str(res)])
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(input_original, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image,people_class_dict[str(res)], (20, 60) , cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255), 2)
    cv2.imshow(people_class_dict[str(res)],expanded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()