import cv2
import streamlit
import time
import tensorflow as tf
import numpy as np
streamlit.title("Action Detection")
streamlit.subheader('Absdiff feed')
FRAME_WINDOW = streamlit.image([])
streamlit.subheader('Live CAM feed')
FRAME_WINDOW1 = streamlit.image([])
camera = cv2.VideoCapture(0)
model2 = tf.keras.models.load_model('C:\\Users\\nithe\\PycharmProjects\\pstreamlit\\model.h5')

while True:
    _, frame1 = camera.read()
    time.sleep(0.1)

    _, frame2 = camera.read()
    time.sleep(0.1)


    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (256, 256))
    image1=  np.dstack([image1]*3)
    image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2, (256, 256))
    image2 = np.dstack([image2]* 3)
    absdiff = cv2.absdiff(image1,image2)

    absdiff1 = np.expand_dims(absdiff, axis=0)


    a=model2.predict(absdiff1)
    if a==0:
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (0, 30)

        # fontScale
        fontScale = 0.5

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        absdiff = cv2.putText(absdiff, 'Signed', org, font,
                            fontScale, color, thickness, cv2.LINE_AA)


    FRAME_WINDOW.image(absdiff,width=500)

    FRAME_WINDOW1.image(frame1,width=500)






else:
    streamlit.write('Stopped')
