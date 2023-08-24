import cv2
import os
import numpy as np
import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)*2 + (y2 - y1)*2)

def road_width(original,image):
    red = [0,0,128]
    Y, X = np.where(np.all(image==red,axis=2))
    Y_axies=1100
    check=1
    set_width=-4
    add_width=  set_width#for DDPI -5 and for Aaksha 8
    x_set=0
    while True:
        list=[]
        for i,y in enumerate(Y):

            if Y[1]>Y_axies:
                Y_axies+=50
                x_set=60
            if y==Y_axies:
                if not((X[i]<(60-x_set))  or (X[i]>(2500+x_set))):
                    list.append(i)
                    check=0
                else:
                    check=1
                    break
            
        if check==0:
            break
        Y_axies-=50
        add_width+=(abs(set_width)//2)

        

    x1_cor=list[0]
    x2_cor=list[-1]
    cv2.line(original, (X[x1_cor], Y_axies), (X[x2_cor], Y_axies), (0, 255, 0), 2)
    width=calculate_distance(X[x1_cor], 900,X[x2_cor], 900)
    road_width=((width/100))+add_width 
    cv2.putText(original, f"Road Width: {road_width:.2f} -- Y = {Y_axies}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return original
