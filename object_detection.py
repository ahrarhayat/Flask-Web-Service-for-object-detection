# import the necessary packages
import base64
import json
from flask import Flask, jsonify, request

app = Flask(__name__)
import numpy as np
import sys
import time
import cv2
import os

# construct the argument parse and parse the arguments
confthres = 0.3
nmsthres = 0.1


def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    lpath = os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath


def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath


def load_model(configpath, weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def do_prediction(image, net, LABELS):
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # TODO Prepare the output as required to the assignment specification(DONE)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        # create an array of the results
        result = []
        #loop over the objects and put them in an array and return the result
        for i in idxs.flatten():
            result.append({"label": LABELS[classIDs[i]], "accuracy": confidences[i],
                           "rectangle": {"height": boxes[i][0], "left": boxes[i][1], "top": boxes[i][2],
                                         "width": boxes[i][3]}})
        return result


## argument
if len(sys.argv) != 2:
    raise ValueError("Argument list is wrong. Please use the following format:  {} {}".
                     format("python object_detection.py", "<yolo_config_folder>"))

yolo_path = str(sys.argv[1])

## Yolov3-tiny versrion
labelsPath = "coco.names"
cfgpath = "yolov3-tiny.cfg"
wpath = "yolov3-tiny.weights"

Lables = get_labels(labelsPath)
CFG = get_config(cfgpath)
Weights = get_weights(wpath)


# TODO, you should  make this console script into webservice using Flask(DONE)
#The main method is made into the starting point for the webservice
@app.route("/api/object_detection", methods=['POST'])
def process_image():
    #check if the incoming message is a HTML POST message, if so, go forward and process the image
    try:
        if request.method == 'POST':
            #get the json data
            req_body = request.get_json()
            #convert the json data
            body_data = json.loads(req_body)
            #extract the id from the data
            id = body_data['id']
            #get the base64 encoded image
            json_image_data = body_data['image']
            #decode the image and save as a file called temp.jpg
            image_data = base64.b64decode(json_image_data)
            filename = 'temp.jpg'
            with open(filename, 'wb') as f:
                f.write(image_data)
                f.close()
            #now that the image is extracted, use the same methods as before and use the methods to get the objects
            img = cv2.imread(filename)
            npimg = np.array(img)
            image = npimg.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # load the neural net.  Should be local to this method as its multi-threaded endpoint
            nets = load_model(CFG, Weights)
            result = do_prediction(image, nets, Lables)
            #finally using jsonify, create json data with an id and array of objects and send back to Client
            return jsonify(id=id, objects=result)

    except Exception as e:

        print("Exception  {}".format(e))


if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port='5006')
