import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import PIL.ImageOps
import os,ssl,time
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if(not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

#to fetch data from the OpenMl library
x, y = fetch_openml("mnist_784", version = 1, return_X_y = True)

#to display the count of each sample
#print(pd.Series(y).value_counts())

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
n = len(classes)

x_train, x_test, y_train, y_test = tts(x, y, random_state=3, train_size=7500, test_size=2500)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

model = LogisticRegression(solver = 'saga', multi_class='multinomial').fit(x_train_scaled, y_train)
prediction = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, prediction)

print(accuracy)

cam = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))

        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
        roi = gray[upper_left[1]: bottom_right[1], upper_left[0], bottom_right[0]]

        pil_image = Image.fromarray(roi)
        img = pil_image.convert("L")
        image_inv = PIL.ImageOps.invert(img)
        pixel_filter = 20

        minimum_pixel = np.percentile(image_inv, pixel_filter)
        image_scaled = np.clip(image_inv-minimum_pixel, 0, 255)
        maximum_pixel = np.max(image_inv)
        image_scaled = np.asarray(image_scaled)/maximum_pixel

        test_sample = np.array(image_scaled).reshape(1, 784)
        predictiion = model.predict(test_sample)

        print(predictiion)

        cv2.imshow("frame", gray)
        
        if(cv2.waitKey(1) & 0xFF == ord("Q")):
            break

    except Exception as e:
        pass

cam.release()
cv2.destroyAllWindows()