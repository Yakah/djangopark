from django.conf import settings
from tensorflow.keras.applications.mobilenet import preprocess_input # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from imutils.video import VideoStream, FPS
import cv2, os, urllib.request
import numpy as np
import pickle
from smartpark.settings import BASE_DIR
from pyml.src.utils import Park_classifier
from .models import Video


        
# car park slot clasifier instead of importing from external file can be build here as a class

# The following commented code was to additionally detect human faces on the parking lot using the haarcascade model, although yolo should be tried since haarcascade loses on accuracy 


#loading serialized face detector model from disk. see how to load cars model from yolov8
# face_detection_videocam = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))



#load the serialized face detector model from disk. to be replaced by car detection model or YOLOV8 model
# prototxtPath = os.path.join(os.path.dirname(BASE_DIR), "ptmodels/face_detector/deploy.prototxt")
# weigthsPath = os.path.join(os.path.dirname(BASE_DIR),"ptmodels/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
# # prototxtPath = os.path.join([settings.BASE_DIR,"ptmodels\\face_detector\\deploy.prototxt"])
# # weigthsPath = os.path.join([settings.BASE_DIR,"ptmodels\\face_detector\\res10_300x300_ssd_iter_140000.caffemodel"])

# # facenet is to detect faces, to be replaced by an array of parking slots to detect white separator lines
# faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weigthsPath)
# maskNet = load_model(os.path.join(settings.BASE_DIR,'ptmodels/face_detector/mask_detector.model'))


# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
        
#     def __del__(self):
#         self.video.release()
        
#     def get_frame(self):
#         success, image = self.video.read()
        
#     # note that we are using motion JPEG but openCV defaults to capture raw images
#     # thus it is recommended we encode the raw images into JPEG in order to correctly display the video stream as follows
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces_detected = face_detection_videocam.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
#         # this faces_detected is to be replaced by slots_detected
#         # the following for loop is to draw a red rectangle around the face objects, to mark the boundaries. to be replaced by slot dimenstions ie trained model
#         for(x,y,w,h) in faces_detected:
#             cv2.rectangle(image, pt1=(x,y),pt2=(x+w,y+h), color=(255, 0, 0), thickness=2)
#         frame_flip = cv2.flip(image, 1)
#         ret, jpeg = cv2.imencode('jpg', frame_flip)
#         return jpeg.tobytes()

# # Reading a video file saved in the local storage
class VideoFeed(object):
    from .utils import Park_classifier
    def __init__(self):
        # defining the params
        rect_width, rect_height = 107, 48
        # carp_park_positions_path = "ptmodels/CarParkPos"
        car_park_positions_path = os.path.join(settings.BASE_DIR,"ptmodels/CarParkPos")
        video_path = "media/videos/carPark.mp4"
        self.vs = VideoStream(src=video_path).start()
        self.fps = FPS().start()

        # creating the classifier  instance which uses image processes to classify
        classifier = Park_classifier(car_park_positions_path, rect_width, rect_height)
        self.video = cv2.VideoCapture('media/videos/carPark.mp4')
    
        
    def __del__(self):
        # self.video.release()
        self.video.release()
        
    def get_frame(self):
        # success, image = self.video.read()
        rect_width, rect_height = 107, 48
        frame = self.vs.read()
        car_park_positions_path = os.path.join(settings.BASE_DIR,"ptmodels/CarParkPos")
        classifier = Park_classifier(car_park_positions_path, rect_width, rect_height)
        # processing the image frames to prepare classify
        processed_frame = classifier.implement_process(frame)
        
        # drawing car park slots according to its status
        denoted_image = classifier.classify(image=frame, prosessed_image=processed_frame)
        
        # saving a screen shot if letter s is pressed
        k = cv2.waitKey(1)
        if k & 0xFF == ord('s'):
            cv2.imwrite("incident.jpg", denoted_image)
        
        return denoted_image.tobytes()
        
    # note that we are using motion JPEG but openCV defaults to capture raw images
    # thus it is recommended we encode the raw images into JPEG in order to correctly display the video stream as follows
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # faces_detected = face_detection_videocam.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        # # # this faces_detected is to be replaced by slots_detected
        # # # the following for loop is to draw a red rectangle around the face objects, to mark the boundaries. to be replaced by slot dimenstions ie trained model
        # # for(x,y,w,h) in faces_detected:
        # #     cv2.rectangle(image, pt1=(x,y),pt2=(x+w,y+h), color=(255, 0, 0), thickness=2)
        # frame_flip = cv2.flip(image, 1)
        # ret, jpeg = cv2.imencode('jpg', frame_flip)
        # return jpeg.tobytes()



#the following class is to alternatively capture live streaming from an IP webcam 
# class IPWebCam(object):
#     def __init__(self):
#         self.url = "http://192.168.0.100/8080/shot.jpg"
        
#     def __del__(self):
#         cv2.destroyAllWindows()
        
#     def get_frame(self):
#         imgResp = urllib.request.urlopen(self.url)
#         imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
#         img = cv2.imdecode(imgNp, -1)
        
#         # like the videoclass above, we are using motion JPEG but openCV by defalt captures raw images
#         # therefore we have to encode it into JPEG so as to correctly display the video stream
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces_detected = face_detection_videocam.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
#         # this faces_detected is to be replaced by slots_detected
#         # the following for loop is to draw a red rectangle around the face objects, to mark the boundaries. to be replaced by slot dimenstions ie trained model
#         for(x,y,w,h) in faces_detected:
#             cv2.rectangle(img, pt1=(x,y),pt2=(x+w,y+h), color=(255, 0, 0), thickness=2)
#         resize = cv2.resize(img, (640,480), interpolation=cv2.INTER_LINEAR)
#         frame_flip = cv2.flip(resize, 1)
#         ret, jpeg = cv2.imencode('jpg', frame_flip)
#         return jpeg.tobytes()

# # class to detect a mask on the video footage. to edit it to detect the parking slots and mask them
# class MaskDetect(object):
#     def __init__(self):
#         self.vs = VideoStream(src=0).start()
        
#     def __del__(self):
#         cv2.destroyAllWindows()
        
#     def detect_and_predict_mask(self, frame, faceNet, maskNet):
#         # grab the dimensions of the frame and then construct a blob from it.
#         # in the case of a parking slot, draw a blob around the white slot markers
#         (h,w) = frame.shape[:2]
#         # the blob below should be of color green, and in out case, should detect a vehicle entering the parking slot and disappear when it leaves
#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))
        
#         # pass the blob through the network and obtain the face detections/car detections
#         faceNet.setInput(blob)
#         detections = faceNet.forward()
        
#         # initialize the list of faces/cars, thrir coresponding locations and list of predictions from the facemask /carmask network
#         # faces should be replaced by cars
#         faces = []
#         locs = []
#         preds = []
#         img_to_array = []
        
#         # run a loop over detections
#         for i in range(0, detections.shape[2]):
#             # extract the confidence, that is probability associated with the detection
#             confidence = detections[0, 0, i, 2]
            
#             # filter out weak detections by ensuring that the confidence is greater than the minimum which is 0.5
#             if confidence > 0.5:
#                 # compute the (x,y) coordinates of the bounding box for the object
#                 box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
#                 (startX, startY, endX, endY) = box.astype("int")
                
#                 # fine tune the bounding boxes to fall within the dimenstions of the frame
#                 (startX, startY) = (max(0, startX), max(0,startY))
#                 (endX, endY) = (min(w-1, endX), min(h-1,endY))
                
#                 # extract the face/ca ROI, convert it from BGR2RGB channel, ordering and resizing it to 224x24 and process it
#                 face = frame[startY:endY, startX:endX]
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#                 face = cv2.resize(face,(224, 224)) #this size is subject to change
#                 face = img_to_array(face)
#                 face = preprocess_input(face)
                
#                 # add the face and bounding boxes to their respective lists
#                 faces.append(face)
#                 locs.append((startX, startY, endX, endY))
                
#         # make a prediction only if at least one face/car was detected
#         if len(faces) > 0:
#             # for faster inference i will make the batch predictions on *all* faces at the same time instead of one by one in the above loop
#             faces = np.array(faces, dtype="float32")
#             preds = maskNet.predict(faces, batch_size = 32)
            
#         # then return a 2-tuple of the face locations and their corresponding locations
#         return(locs, preds)
    
#     def get_frame(self):
#         frame = self.vs.read()
#         frame = imutils.resize(frame, width = 650)
#         frame = cv2.flip(frame, 1)
#         # detet faces in the frame and dtermine if they are 
#         (locs, preds) = self.detect_and_predict_mask(frame,faceNet, maskNet)
        
#         # loop over detected face locations and their corresponding locations
#         for (box, pred) in zip(locs, preds):
#             # unpack the bounding box and predictions
#             (startX, startY, endX, endY) = box
#             (mask, withoutMask) = pred
            
#             # determne the class label and color we wll use to draw the bounding box and text
#             label = "Mask" if mask > withoutMask else "No Mask"
#             color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
#             # include the probability in the label
#             label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
#             # display the label and bounding box rectangle on the output frame
#             cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
#             cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         return jpeg.tobytes()
    
# # to get a feed from live CCTV camera connected to the computer via the rstp protocol 
# class LiveWebcam(object):
#     def __init__(self):
#         # to capture the footage from IP webcam
#         self.url = cv2.VideoCapture("rstp://admin:pass@123@192.168.0.2:554/")
        
#     # method to end the session capture
#     def __del__(self):
#         cv2.destroyAllWindows()
        
#     # function to process the footage
#     def get_frame(self):
#         success, imgNp = self.url.read()
#         resize = cv2.resize(imgNp, (640, 480), interpolation=cv2.INTER_LINEAR)
#         ret, jpeg = cv2.imencode('.jpg', resize)
#         return jpeg.tobytes()
                
