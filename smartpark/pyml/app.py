import cv2
import numpy as np
import pickle
from src.utils import Park_classifier


def demostration():
    """It is a demonstration of the application.
    """

    # defining the params
    rect_width, rect_height = 107, 48
    carp_park_positions_path = "data/source/CarParkPos"
    video_path = "data/source/carPark.mp4"

    # creating the classifier  instance which uses image processes to classify
    classifier = Park_classifier(carp_park_positions_path, rect_width, rect_height)

    # Implementation of the classy
    cap = cv2.VideoCapture(video_path)
    frameCounter = 0
    while True:

        # reading the video frame by frame
        ret, frame = cap.read()
        frameCounter += 1
        
        # if the last frame of the video is reached, reset the counter to enable the video to loop
        if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frameCounter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)

        # check is there a retval
        if not ret:break
        
        # prosessing the frames to prepare classify
        prosessed_frame = classifier.implement_process(frame)
        
        # drawing car parks according to its status 
        denoted_image = classifier.classify(image=frame, prosessed_image = prosessed_frame)
        
        # displaying the results
        cv2.imshow("Empty and occupied Slots", denoted_image)
        
        # exit condition
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        
        if k & 0xFF == ord('s'):
            cv2.imwrite("output.jpg", denoted_image)

    # re-allocating sources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demostration()
