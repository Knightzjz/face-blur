# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import sys
import time
import numpy as np
import cv2

import face


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            '''
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
                          '''
            if face.name == 'Xiao':
                cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (255, 255, 0), 2)
                
                cv2.putText(frame, 'Xiao', (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                            thickness=2, lineType=2)
                
                #frame=face.blurred
                





                '''
                face=cv2.GaussianBlur(face.image,(15,15),0)
                
                crop_blur = frame[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
                frame=cv2.GaussianBlur(crop_blur,(5,5),0)


                
                print crop_blur
                
                cv2.
                np.minimum(crop_blur,0)
                '''
                
                '''
                cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), -1)
                '''

                '''
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)
                
            else:
                cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (255, 255, 0), 2)
                
                cv2.putText(frame, 'Xiao', (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                            thickness=2, lineType=2)
            '''    
            #take the blurred face and place it on the original frame, the blurred area will
            #be subsitute by the blurred image since both of them are the same size.
            else: 
                frame[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2],
                                                 :]=cv2.GaussianBlur(face.blurred, (25,25),0)

                


    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)
    #cv2.imshow('Intern Video', frame) 

def main(args):
    frame_interval = 1  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    
    video_capture = cv2.VideoCapture("test4.mp4")
    face_recognition = face.Recognition()
    start_time = time.time()
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (int(video_capture.get(3)),int(video_capture.get(4))),True)



    if args.debug:
        print("Debug enabled")  
        face.debug = True

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        
            
        
        add_overlays(frame, faces, frame_rate)

        frame_count += 1
        #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #cv2.imshow('gray',gray)
        out.write(frame)
            #print 'a'
        



        cv2.imshow('Vedio', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
