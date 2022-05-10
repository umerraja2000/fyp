from asyncio.windows_events import NULL
from pyexpat import model
from flask import Flask, render_template, Response, request
import cv2
from keras_preprocessing.image import ImageDataGenerator
import os, sys
import numpy as np
from PIL import Image
import numpy as np
from keras.preprocessing import image


global capture,switch
capture=0
switch=1

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

from keras.models import load_model
classifier = load_model('Trained_model2.h5')

test_path = './mydata/asl_alphabet_test/asl_alphabet_test/'
train_path= './mydata/asl_alphabet_train/asl_alphabet_train/'
datagen = ImageDataGenerator(rescale=1/255, validation_split=0.3)

train = datagen.flow_from_directory(train_path, subset='training')


def predictor():
      from keras.preprocessing import image
      for i in enumerate(os.listdir(test_path)):
        image = Image.open('2.jpg')#+'/'+i[1])
        image.resize((200,200), resample=0)
        image = np.asarray(image)
        image = image/255
    
        pred = np.argmax(classifier.predict(image.reshape(-1,200,200,3)))
        for j in train.class_indices:
            if pred == train.class_indices[j]:
                prediction=j
            else:
                continue
        return prediction


def gen_frames():  # generate frame by frame from camera
    global out, capture , img_text
    img_text=" "
    
    while True:
        success, frame = camera.read() 
        frame = cv2.flip(frame,1)
        if success:
            
            if(capture):
                capture=0
                img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)
                
                imcrop = img[102:298, 427:623]
                
                p = "2.jpg"
                save_img = cv2.resize(imcrop, (200, 200))
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