import Detector
import cv2 as cv
import numpy as np
import streamlit as st
import pandas as pd

import time

# Bib bounding box color
color = [252, 15, 192]
# Initialize rank order list
rank = []

# App Title and Mode
st.title('Race Bib Number Detector')
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    'Select Mode',
    options=['Demo', 'Image', 'Video']
)

if mode == 'Image':
    # Get Image from User
    user_file = st.file_uploader(label='Image for analysis:',
        type=['jpg', 'png'])

    button_loc = st.empty()
    
    # Display image and convert for predicting
    if user_file != None:
            text_loc = st.empty()
            text_loc.text('This is your image:')
            # Convert the file to an opencv image.
            # From: https://github.com/streamlit/streamlit/issues/888
            file_bytes = np.asarray(bytearray(user_file.read()), dtype=np.uint8)
            img = cv.imdecode(file_bytes, 1)
            img_loc = st.empty()
            img_loc.image(img, channels='BGR')

            if(button_loc.button('Detect')):

                # get bib prediction
                start = time.time() # start timing prediction
                output = Detector.get_bibs(img)
                end = time.time() # end timing prediction

                for detection in output:
                    # draw bouding box on original image
                    (x, y, w, h) = detection[1]
                    img = cv.rectangle(img,(x,y),(x+w,y+h),color,5)
                    # add bib number to original image
                    rbn = detection[0]
                    cv.putText(img, str(rbn), (x, y - 25), cv.FONT_HERSHEY_SIMPLEX, 2, color, 4)

                #display detection time and annotated image
                text_loc.text(f'Detection time: {round(end - start, 2)} seconds')
                img_loc.image(img, channels='BGR')
else:
    if mode == 'Demo':
        video_file = open('../Data/bib_detector_demo_edit.mp4', 'rb')
    elif mode == 'Video':
        video_file = st.file_uploader(label='Video for analysis:',
        type=['mp4'])

    button_loc = st.empty()
    text_loc = st.empty()
    video_loc = st.empty()

    if video_file != None:
        video_bytes = video_file.read()
        video_loc.video(video_bytes)

        if(button_loc.button('Detect')):
            cap = cv.VideoCapture('../Data/bib_detector_demo_edit.mp4')
            cap.set(cv.CAP_PROP_FPS, 25)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # get bib prediction
                start = time.time() # start timing prediction
                output = Detector.get_bibs(frame, single=True)
                end = time.time() # end timing prediction

                for detection in output:
                    # draw bouding box on original image
                    (x, y, w, h) = detection[1]
                    img = cv.rectangle(frame,(x,y),(x+w,y+h),color,5)
                    # add bib number to original image
                    rbn = detection[0]
                    cv.putText(frame, str(rbn), (x, y - 25), cv.FONT_HERSHEY_SIMPLEX, 2, color, 4)

                #display detection time and annotated image
                text_loc.text(f'Detection time: {round(end - start, 2)} seconds')
                video_loc.image(frame, channels='BGR')