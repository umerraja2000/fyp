from asyncio.windows_events import NULL
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread


global capture,switch
capture=0
switch=1

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

from keras.models import load_model
classifier = load_model('Trained_model.h5')
def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'A'
       elif result[0][1] == 1:
              return 'B'
       elif result[0][2] == 1:
              return 'C'
       elif result[0][3] == 1:
             return 'D'
       elif result[0][4] == 1:
             return 'E'
       elif result[0][5] == 1:
             return 'F'
       elif result[0][6] == 1:
             return 'G'
       elif result[0][7] == 1:
             return 'H'
       elif result[0][8] == 1:
             return 'I'
       elif result[0][9] == 1:
             return 'J'
       elif result[0][10] == 1:
             return 'K'
       elif result[0][11] == 1:
              return 'L'
       elif result[0][12] == 1:
              return 'M'
       elif result[0][13] == 1:
             return 'N'
       elif result[0][14] == 1:
             return 'O'
       elif result[0][15] == 1:
             return 'P'
       elif result[0][16] == 1:
             return 'Q'
       elif result[0][17] == 1:
             return 'R'
       elif result[0][18] == 1:
             return 'S'
       elif result[0][19] == 1:
             return 'T'
       elif result[0][20] == 1:
             return 'U'
       elif result[0][21] == 1:
              return 'V'
       elif result[0][22] == 1:
             return 'W'
       elif result[0][23] == 1:
             return 'X'
       elif result[0][24] == 1:
             return 'Y'
       elif result[0][25] == 1:
              return 'Z'


def gen_frames():  # generate frame by frame from camera
    global out, capture , img_text
    
    while True:
        success, frame = camera.read() 
        frame = cv2.flip(frame,1)
        if success:
           
            if(capture):
                capture=0
                img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)
                lower_blue =0
                upper_blue =0
                imcrop = img[102:298, 427:623]
                
                hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)
                (thresh, hsv) = cv2.threshold(hsv, 127, 255, cv2.THRESH_BINARY)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                p = "1.png"
                save_img = cv2.resize(mask, (64, 64))
                cv2.imwrite(p, save_img)
                img_text = predictor()
                print(img_text)
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
        

@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
       
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
             
    elif request.method=='GET':
        return render_template('index.html' )
    return render_template('index.html' ,  prediction = img_text)


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     
