Requirements:

tensorflow 2.14.0
opencv-python 4.8.1.78
keras 2.14.0
mtcnn 0.1.1

The validate data provided has already been processed for direct verification.
The original labelled images were initally scanned for faces using both haar
cascades and a pretrained mtcnn arichitecture for faces. mtcnn albiet slower
provided much better results compared to haar cascades.

The  fine tuned model has been trained on those images.

To run the program:

For verification:
    Execute - Face_Recognizer.py :This script valides the model on a new set of data
of 64 images.
Another 20 images numbered from 1 to 20 are provided for manual verification.
To add more validation images please add detected faces to validate folder 
named to convention. Relative paths are for the folders. Please execute the file
in the folder that is present.

Training_model.py was used to train the model and the model was saved to file
face_recognition.h5. 


Face_train2.py was used to preprocess the data. Face_train2.py used mtcnn face detection
model to find faces in from the orginal test data that was downloaded from 
https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset. The images where  
searched for a face and the face's bounding box was cropped out and stored to be used for
training. Face_train2.py can't be run because the orginial data set is not submitted.