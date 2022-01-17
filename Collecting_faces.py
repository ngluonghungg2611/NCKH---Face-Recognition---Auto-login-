import cv2
import numpy as np
import os
# from face_detection_ssd import draw_fancy_box
import draw_fancy_box
background = None
accumulated_weight = 0.5
ROI_top = 130
ROI_bottom = 380
ROI_right = 380
ROI_left = 130

face_cascade_opposite = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
cam = cv2.VideoCapture(0)
num_frames= 0
num_image_takens = 0
face_opposite = 0
face_left = 0
face_right = 0
face_look_up = 0
face_look_down = 0
name = input('Import name of  user: ')

while name in os.listdir('./images'):
# while name in os.listdir('D:\\COmputerVision\\NCKH\\image'):
    name = input('Name already exists. Import again: ')
else:
    os.makedirs('images/' + name)
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)   
        frame_faces = frame.copy()          
        roi = frame[ROI_top:ROI_bottom, ROI_left:ROI_right]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # gray_roi = cv2.GaussianBlur(gray_roi, (9,9), 0)
        faces_opposite = face_cascade_opposite.detectMultiScale(frame, 1.3, 5)
        faces_alt = face_cascade_alt.detectMultiScale(frame, 1.3, 5)
        # Chuan bi background
        if num_frames < 120:
            cv2.putText(frame, 'Checking Background!', (50, 430), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)
            cv2.putText(frame, 'Please Wait: ' + str((num_frames // 30)+1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)
        
        
        #------------------------------------------ Yeu cau lay anh truc dien------------------------------------
        elif num_frames >= 120 and num_frames < 200:
            cv2.putText(frame, "Get your opposite face in the", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "bounding box! ", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "Please wait... " + str((num_frames - 120) // 60 + 1), (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        #   Lay anh truc dien    
        elif num_frames >= 200 and num_frames < 300:
            for (x,y,w,h) in faces_opposite:
                if x > ROI_left & y > ROI_bottom & w < (ROI_left + ROI_right) & h < (ROI_top + ROI_bottom) & x+w < ROI_right: # Dieu kien nam trong bounding box
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
                    draw_fancy_box.draw_fancy_box(frame, (x,y), (x+w,y+h), (127, 255, 255), 2, 10, 20)
                    cv2.putText(frame, "Invalid Face...Get Opposite face", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    # cv2.putText(frame, "Count image: " + str(face_opposite), (50,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)
                    img_faces = cv2.resize(frame_faces[y:y+h, x:x+w], (224,224))
                    cv2.imshow('Opposite face', img_faces)
                    cv2.imwrite(r'images/' + name + '/' + '1' '.jpg', img_faces)

                else:
                    cv2.putText(frame, "Get your face into the", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
                    cv2.putText(frame, "bounding box(A far, outside)", (30,90), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
                


        # -------------------------------------------- Yeu cau lay anh mat trai--------------------------------------------
        elif num_frames >= 300 and num_frames < 400:
            cv2.putText(frame, "Get your left face in the", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "bounding box! ", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "Please wait... " + str((num_frames - 300) // 60 + 1), (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        # Lay anh mat trai
        elif num_frames >= 400 and num_frames < 500:      
            for (x,y,w,h) in faces_alt:
                if x > ROI_left & y > ROI_bottom & w < (ROI_left + ROI_right) & h < (ROI_top + ROI_bottom) & x+w < ROI_right:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
                    draw_fancy_box.draw_fancy_box(frame, (x,y), (x+w,y+h), (127, 255, 255), 2, 10, 20)
                    cv2.putText(frame, "Invalid Face...Get left side face", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    # cv2.putText(frame, "Count image: " + str(face_left), (50,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)
                    img_faces = cv2.resize(frame_faces[y:y+h, x:x+w], (224,224))
                    cv2.imshow('Left side face', img_faces)                        
                    cv2.imwrite(r'images/' + name + '/' + '2' '.jpg', img_faces)
                else:
                    cv2.putText(frame, "Get your face into the", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                    cv2.putText(frame, "bounding box !", (50,90), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
                    
                    
                    
        #----------------------------------------------- Yeu cau lay anh mat phai-------------------------------------
        elif num_frames >= 500 and num_frames<600:
            cv2.putText(frame, "Get your right face in the", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "bounding box! ", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "Please wait... " + str((num_frames - 500) // 60 + 1), (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        elif num_frames >= 600 and num_frames < 700:
            for (x,y,w,h) in faces_alt:
                if x > ROI_left & y > ROI_bottom & w < (ROI_left + ROI_right) & h < (ROI_top + ROI_bottom) & x+w < ROI_right:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
                    draw_fancy_box.draw_fancy_box(frame, (x,y), (x+w,y+h), (127, 255, 255), 2, 10, 20)
                    cv2.putText(frame, "Invalid Face...Get right side face", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    # cv2.putText(frame, "Count image: " + str(face_right), (50,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)
                    img_faces = cv2.resize(frame_faces[y:y+h, x:x+w], (224,224))
                    cv2.imshow('Right right side face', img_faces)
                    cv2.imwrite(r'images/' + name + '/' + '3' '.jpg', img_faces)
                
                else:
                    cv2.putText(frame, "Get your face into the", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                    cv2.putText(frame, "bounding box !", (50,90), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        
        # --------------------------------------------Yeu cau lay anh nhin le tren -----------------------------------------
        elif num_frames >= 700 and num_frames < 800:
            cv2.putText(frame, "Get your look up face in the", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "bounding box! ", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "Please wait... " + str((num_frames - 700) // 60 + 1), (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        elif num_frames >= 800 and num_frames < 900:
            for (x,y,w,h) in faces_opposite:
                if x > ROI_left & y > ROI_bottom & w < (ROI_left + ROI_right) & h < (ROI_top + ROI_bottom) & x+w < ROI_right:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
                    draw_fancy_box.draw_fancy_box(frame, (x,y), (x+w,y+h), (127, 255, 255), 2, 10, 20)
                    cv2.putText(frame, "Invalid Face...Get look up face", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    # cv2.putText(frame, "Count image: " + str(face_look_up), (50,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)
                    img_faces = cv2.resize(frame_faces[y:y+h, x:x+w], (224,224))
                    cv2.imshow('Look look up face', img_faces)
                                             
                    cv2.imwrite(r'images/' + name + '/' + '4' '.jpg', img_faces)                        
                        
                else:
                    cv2.putText(frame, "Get your face into the", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                    cv2.putText(frame, "bounding box !", (50,90), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)    
                    
                    
                    
        
        # --------------------------------------------Yeu cau lay anh nhin xuong duoi ------------------------------------------
        elif num_frames >= 900 and num_frames < 1000:
            cv2.putText(frame, "Get your look down face in the",(50, 50),  cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "bounding box! ", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.putText(frame, "Please wait... " + str((num_frames - 900) // 60 + 1), (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        elif num_frames >= 1000 and num_frames < 1100:
            for (x,y,w,h) in faces_opposite:
                if x > ROI_left & y > ROI_bottom & w < (ROI_left + ROI_right) & h < (ROI_top + ROI_bottom) & x+w < ROI_right:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
                    draw_fancy_box.draw_fancy_box(frame, (x,y), (x+w,y+h), (127, 255, 255), 2, 10, 20)
                    cv2.putText(frame, "Invalid Face...Get look down face", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    # cv2.putText(frame, "Count image: " + str(face_look_down), (50,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)
                    img_faces = cv2.resize(frame_faces[y:y+h, x:x+w], (224,224))
                    cv2.imshow('Look down face',img_faces)                        
                    cv2.imwrite(r'images/' + name + '/' + '5' '.jpg', img_faces)
                    
                else:
                    cv2.putText(frame, "Get your face into the", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                    cv2.putText(frame, "bounding box !", (50,90), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)    
        
        # elif num_frames >= 1100 and num_frames < 1200:
        else:
            
            cv2.putText(frame, "Well done", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, "Thank you", (50,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
            
        # else:
        #     break
        
        num_frames += 1   
                # print('x: ',x > ROI_left)
                # print('y: ',y > ROI_bottom)
                # print('x + w: ',x + w < ROI_right)
                # print('y + h: ',y + h < ROI_top)
            # cv2.imread('//')
        cv2.rectangle(frame, (ROI_left, ROI_bottom), (ROI_right, ROI_top), (255,255,0), 2)
        
        cv2.imshow('Original camera', frame)
        cv2.imshow('Face detected',roi)
        cv2.imshow('Grayscale face', gray_roi)
        
        
        cv2.waitKey(1)
            
    cam.release()
    cv2.destroyAllWindows()

        

        

    
