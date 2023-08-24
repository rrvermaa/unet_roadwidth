import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from contour import contour
from model import *
MASK=True
ROI=False
if __name__ == "__main__":

    """ Process Video """
    video_path = 'videos/20230523120642_0060.mp4'  
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('videos/demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))

    count=0
    while cap.isOpened():
        ret, image = cap.read()
        count+=1
        print(count)
        if not ret  :
            break
        
        if count%200 == 0:

            # overlay,roi_frame=road_extraction(image,frame_width,frame_height)

            save_image_path = f"results/frame_{count}.png"
            cv2.imwrite(save_image_path, image)

            # out.write(overlay)



    cap.release()
    cv2.destroyAllWindows()

