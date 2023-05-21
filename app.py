from flask import Flask,render_template,request,redirect,url_for
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

app.config['INPUT_IMAGE_FILE_PATH'] = 'C:/Users/mathesh/Documents/Python/flask_Learning/New_proj/Input_Images/'

@app.template_filter('zip')
def _zip(a, b):
    return zip(a, b)


@app.route('/')
def hello_world():
    return render_template("index.html")





@app.route('/upload', methods=['POST','GET'])
def upload():
    if request.method=='POST':
        input_image = request.files["file"]
        global input_image_filepath
        input_image_filepath = "static/images/Uploaded_Images/input_image.jpg"
        input_image.save(input_image_filepath)
       
        img=cv2.imread(input_image_filepath)
        model = YOLO("yolo_weights/yolov8l")
        result = model(img)
        global name
        name=[]
        global numbers
        numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        # print(numbers)
        for r in result:
            boxes = r.boxes
            for box in boxes:
                classNames = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
                    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
                    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
                    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
                    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
                }
                

                clsName = int(box.cls[0])
                x1, x2, y1, y2 = box.xyxy[0]
                a, b, c, d = int(x1), int(x2), int(y1), int(y2)
                cv2.rectangle(img, (a, b), (c, d), (0, 0, 255), thickness=2)
                cv2.putText(img, f'{classNames[clsName]}', (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                names=classNames[clsName]
                # print(name)
                
                name.append(names)
                global name_cap
                name_cap=list(map(str.capitalize,name))
                
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        cv2.imwrite("static/images/Output_Images/output_image.jpg",img)
        global output_image_filepath
        output_image_filepath = "static/images/Output_Images/output_image.jpg"
                
        
        
    return render_template("image.html",input_image_filepath=input_image_filepath,names=name_cap ,numbers=numbers , op=output_image_filepath)







if __name__ == '__main__':
    app.run(debug=True)