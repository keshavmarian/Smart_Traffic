
import numpy as np
import cv2
import time
import pathlib


"""
a blueprint for a bounded box with its corresponding name,confidence score and 
"""
print(pathlib.Path.cwd())

class BoundedBox:
    
    def __init__(self, xmin, ymin, xmax, ymax, ids, confidence):
        with open(str(pathlib.Path.cwd().parents[0])+"/AI-based-Traffic-Control-System--/datas/coco.names", 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')  # stores a list of classes
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax       
        self.ymax = ymax
        self.name = self.classes[ids]
        self.confidence = confidence   


"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""

class Lanes:
    def __init__(self,lanes):
        self.lanes=lanes
    
    def getLanes(self):
        
        return self.lanes
    
    def lanesTurn(self):
        
       return self.lanes.pop(0)

    def enque(self,lane):
 
       return self.lanes.append(lane)
    def lastLane(self):
       return self.lanes[len(self.lanes)-1]
"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""
class Lane:
    def __init__(self, count=0, frame=None, lane_number=0):
        self.count = int(count)  # Ensure the count is always an integer
        self.frame = frame
        self.lane_number = lane_number

    
"""
given lanes object return a duration based on comparison of each lane vehicle count
"""
def schedule(lanes):
   
    standard=10 #standard duration
    reward =0  #reward to be added or subtracted on the standard duration
    turn = lanes.lanesTurn()
    
    for i,lane in enumerate(lanes.getLanes()):
        if(i==(len(lanes.getLanes())-1)):
            reward = reward + (turn.count-lane.count)*0.2
        else:
            reward = reward + (turn.count-lane.count)*0.5
    scheduled_time = round((standard+reward),0)
    lanes.enque(turn)
    return scheduled_time
       
"""
given duration and lanes, returns a grid image containing frames of each lane with
their corresponding waiting duration
"""   

def display_result(wait_time, lanes):
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    
    valid_images = []  # List to store valid images for concatenation
    
    for i, lane in enumerate(lanes.getLanes()):
        # Check if the lane has a valid frame before resizing
        if lane.frame is not None:
            # Resized so that all images have the same dimension to be concatenable
            lane.frame = cv2.resize(lane.frame, (1280, 720))

            if wait_time <= 0 and (i == (len(lanes.getLanes()) - 1) or i == 0):
                color = yellow
                text = "yellow:2 sec"
            elif wait_time >= 0 and i == (len(lanes.getLanes()) - 1):
                color = green
                text = "green:" + str(wait_time) + " sec"
            else:
                color = red
                text = "red:" + str(wait_time) + " sec"

            lane.frame = cv2.putText(lane.frame, text, (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 6)
            lane.frame = cv2.putText(lane.frame, "vehicle count:" + str(lane.count), (60, 195), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

            # Add valid frames to the list for later concatenation
            valid_images.append(lane.frame)
        else:
            print(f"Warning: lane {lane.lane_number} has an invalid frame (None), skipping display.")

    # Concatenate the valid frames horizontally and vertically
    if len(valid_images) > 0:
        if len(valid_images) == 1:
            # Only one valid frame, return it directly
            return valid_images[0]
        elif len(valid_images) == 2:
            # Two valid frames, concatenate them horizontally
            return np.concatenate((valid_images[0], valid_images[1]), axis=1)
        elif len(valid_images) == 3:
            # Three valid frames, concatenate the first two horizontally and the third below
            hori_image = np.concatenate((valid_images[0], valid_images[1]), axis=1)
            return np.concatenate((hori_image, valid_images[2]), axis=0)
        else:
            # Four valid frames, concatenate in two rows
            hori_image = np.concatenate((valid_images[0], valid_images[1]), axis=1)
            hori2_image = np.concatenate((valid_images[2], valid_images[3]), axis=1)
            return np.concatenate((hori_image, hori2_image), axis=0)
    
    # If no valid frames, return None
    return None





# given detecteed boxes, return number of vehicles on each box
def vehicle_count(Boxes):
        vehicle=0
        for box in Boxes:
            if box.name == "car" or box.name == "truck" or box.name == "bus":
                vehicle=vehicle+1  

        return vehicle

# given the grid dimension, returns a 2d grid
def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def drawPred( frame, classId, conf, left, top, right, bottom):
        
       
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=6)

        return frame

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    ratioh, ratiow = frameHeight / 320, frameWidth / 320
    classIds = []
    confidences = []
    boxes = []

    # Process the output of YOLO layers
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5 and detection[4] > 0.5:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Apply non-maxima suppression to avoid overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    # Debugging print statement
    print(f"indices: {indices}")
    correct_boxes = []
    
    # Check if indices have any values before accessing them
    if len(indices) > 0:
        for i in indices:
            #i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            box = BoundedBox(box[0], box[1], box[2], box[3], classIds[i], confidences[i])
            correct_boxes.append(box)
            frame = drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
    else:
        print("No valid bounding boxes found.")

    return correct_boxes, frame




"""
interpret the ouptut boxes into the appropriate bounding boxes based on the yolo paper 
logspace transform
"""
def modify(outs,confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        print(str(pathlib.Path.cwd().parents[0])+"/datas")
        with open(str(pathlib.Path.cwd().parents[0])+'/AI-based-Traffic-Control-System--/datas/coco.names', 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')   
        print("dir:"+str(pathlib.Path.cwd()))
        colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(classes))]
        num_classes = len(classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        nl = len(anchors)
        na = len(anchors[0]) // 2
        no = num_classes + 5
        grid = [np.zeros(1)] * nl
        stride = np.array([8., 16., 32.])
        anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, 1, -1, 1, 1, 2)

        
        z = []  # inference output
        for i in range(nl):
            bs, _, ny, nx,c = outs[i].shape  
            if grid[i].shape[2:4] != outs[i].shape[2:4]:
                grid[i] = _make_grid(nx, ny)
                

            y = 1 / (1 + np.exp(-outs[i])) 
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * int(stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1,no))
        z = np.concatenate(z, axis=1)
        return z


"""
given each lanes image, it inferences using trt engine on the image, return lanes object
containg processed image and waiting duration for each image

"""
def final_output_tensorrt(processor,lanes):
     
    for lane in lanes.getLanes():
            lane.frame=cv2.resize(lane.frame,(1280,720))      #resize into a standard image  dimension
            start = time.time()
            output = processor.detect(lane.frame)
            end = time.time() 
            print("fps:"+str(end-start))   
            dets = modify(output)
            boxes,frame = postprocess(lane.frame,dets)
            count = vehicle_count(boxes)
            lane.count= count
            lane.frame=frame
            
        
        
    return lanes

"""
given each lanes image, it inferences onnx model on the image, return lanes object
containg processed image and waiting duration for each image

"""

def final_output(net, output_layer, lanes):
    for lane in lanes.getLanes():
        if lane.frame is not None:
            if isinstance(lane.frame, np.ndarray):
                lane.frame = cv2.resize(lane.frame, (1280, 720))
                blob = cv2.dnn.blobFromImage(lane.frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(output_layer)
                end = time.time()
                print("fps:" + str(end - start))

                dets = modify(layerOutputs)
                boxes, frame = postprocess(lane.frame, dets)
                count = vehicle_count(boxes)
                lane.count = int(count)  # Ensure count is an integer
                lane.frame = frame
            else:
                print(f"Warning: lane {lane.lane_number} frame is not a valid numpy array.")
        else:
            print(f"Warning: lane {lane.lane_number} has an invalid frame (None).")

    return lanes



