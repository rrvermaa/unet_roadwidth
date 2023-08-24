import numpy as np
import cv2
from skimage import exposure 
from width import road_width

# Define the function to get color map
def getMap(frame,cThr=[100,100],minArea=20000,filter=0,draw=True):

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a_component = lab[:,:,1]
    th = cv2.threshold(a_component,140,255,cv2.THRESH_BINARY)[1]
    blur = cv2.GaussianBlur(th,(23,23), 29)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    
    imgCanny= cv2.Canny(blur,cThr[0],cThr[1])
    kernal=np.ones((5,5))
    imgDial= cv2.dilate(imgCanny,kernal,iterations=3)
    imgThr=cv2.erode(imgDial,kernal,iterations=2)
    
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours=[]
    for i in contours:
        area =cv2.contourArea(i)
        if area >minArea:
            peri=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*peri,True)
            bbox=cv2.boundingRect(approx)
            
            if filter >0:
                if len(approx)==filter:
                    finalCountours.append([len(approx),area,approx,bbox,i])
            else:
                finalCountours.append([len(approx),area,approx,bbox,i])

    finalCountours= sorted(finalCountours,key=lambda x:x[1],reverse=True)
    
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, frame, 0.5, 0)
    return super_imposed_img,finalCountours

def contour(original,frame):
        cam,conts = getMap(frame,minArea=900000,draw=False)
        
        map_img = exposure.rescale_intensity(cam, out_range=(0, 255))
        map_img = np.uint8(map_img)
        heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)

        # Merge map and frame
        fin = cv2.addWeighted(heatmap_img, 0.9, frame, 0.9, 0.1)
        print(len(conts))
        if len(conts)!=0:
            biggest=conts[0][2]
            print(biggest)
            cv2.drawContours(original, [biggest], -1, (0, 0, 255), 2)
            
            original=road_width(original)
            

        return original        




