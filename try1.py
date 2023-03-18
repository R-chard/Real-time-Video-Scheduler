import sys
sys.path.insert(1,"./shared/details")
import cv2
import util
import config
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from object_detection import ObjectDetection
from sklearn.linear_model import LinearRegression

# File trains a multi regressoin model
vid_file = "full_traffic.mp4"
capture = cv2.VideoCapture(os.path.join(config.MEDIA_DIR, vid_file))
print("Initializing stream " + vid_file + " of length " + str(capture.get(cv2.CAP_PROP_FRAME_COUNT)) + " frames.")
size_to_disruption = []
speed_to_disruption= []
confidence_to_disruption = []
occlusion_percent_to_disruption = []
prevPos = {}
trackers = []

frame_details = {} 

od = ObjectDetection()
data_points = []
frame_id = 0

def plot_result(points):
    disruption = [0 for _ in range(len(points))]
    sizes = [0 for _ in range(len(points))]
    speeds = [0 for _ in range(len(points))]
    confidences = [0 for _ in range(len(points))]
    occlusion_percents = [0 for _ in range(len(points))]
    xs = [0 for _ in range(len(points))]
    ys = [0 for _ in range(len(points))]

    for i in range(len(points)):
        disruption[i] = points[i][0]
        sizes[i], speeds[i], confidences[i],occlusion_percents[i] = points[i][1]

    y_points = np.array(disruption)

    label = ["Size of object", "Speed of object", "Confidence of DNN detection", "Percentage of object occluded"]
    title = ["size", "speed", "confidence", "occlusion"]
    x_points = [sizes, speeds, confidences, occlusion_percents]
    for i in range(len(x_points)):
        x = np.array(x_points[i])
        gradient, c = np.polyfit(x,y_points,1)

        plt.ylabel("MOSSE prediction accuracy")
        plt.xlabel(label[i])
        plt.scatter(x, y_points)
    
        print("Prediction results for: " + title[i])
        print("Gradient calculated to be: " + str(gradient))
        print("Y-Intercept calculated to be: " + str(c) + "\n")

        file_path = os.path.join(config.SHARED_DIR, config.RESULTS_DIR, title[i] + ".png")
        plt.savefig(file_path)
        plt.clf()
    
def build_regression_model(points):
    tracked_iou, factors = zip(*points)
    tracked_iou = np.array(tracked_iou)
    factors = np.array(factors)
    model = LinearRegression()
    reg = model.fit(factors,tracked_iou)

    print("Factors arranged in order of: Speed, Size, Confidence")
    print("Coefficient of regression model: " + str(reg.coef_))
    print("Score of regression model: " + str(reg.score(factors, tracked_iou)))

    config.save_regression_model(model)

while capture.isOpened():
    success,frame = capture.read()
    if not success:
        break
    dim = config.get_resized_dim(frame)
    frame = cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)
    screen_height, screen_width, _ = frame.shape
    obj_ids, confidences, bboxes = od.detect(frame)

    already_tracked = set()
    existing_trackers = []
    frame_bboxes = []

    print("Training on frame: " + str(frame_id))

    count = 0
    totals = [0,0,0,0]

    crowd_x = 0
    crowd_y = 0

    for i in range(len(trackers)):
        _,tracked_bbox = trackers[i].update(frame)
        if not tracked_bbox[2] and not tracked_bbox[3]:
            tracked_bbox = prevPos[trackers[i]][0]

        for j in range(len(bboxes)):
            if j in already_tracked:
                continue

            # Renaming
            obj_id = obj_ids[j].item()
            confidence = round(confidences[j].item(),3)
            bbox = tuple(bboxes[j])

            max_iou = 0
            max_j = -1

            # Update tracker if DNN returns a slightly different location
            for j in range(len(bboxes)):

                # Ignore background objects
                obj_id = obj_ids[j].item()
                if j in already_tracked or prevPos[trackers[i]][1] != obj_id:
                    continue
                
                bbox = tuple(bboxes[j])
                iou = util.calcIOU(bbox, tracked_bbox)    
                if iou > max_iou:
                    max_j = j
                    max_iou = iou

        # Found match
        if max_j != -1:
            bbox = tuple(bboxes[max_j])
            obj_id = obj_ids[max_j].item()
            speed_x,speed_y = util.calcSpeed(prevPos[trackers[i]][0], bbox, 1)
            centre_x = bbox[0] + bbox[2]/2
            centre_y = bbox[1] + bbox[3]/2

            if centre_x < screen_width/2:
                crowd_x += speed_x
            else:
                crowd_x -= speed_x

            if centre_y < screen_height/2:
                crowd_y += speed_y
            else:
                crowd_y -= speed_y

            speed = (speed_x**2 + speed_y**2)**0.5
            size = bbox[2] * bbox[3]
            occlusion_percent = util.percent_occluded(bbox,bboxes)

            # update internal structures
            totals[0] += speed
            totals[1] += size
            totals[2] += confidence
            totals[3] += occlusion_percent
            count += 1
            already_tracked.add(max_j)

            prevPos[trackers[i]][0] = bbox
            trackers[i].init(frame,bbox)
            frame_bboxes.append((bbox, obj_id))

            existing_trackers.append(trackers[i])
        else:
            del prevPos[trackers[i]]

    if count == 0:
        data_points.append([0,[0,0,0,0]])
    else:
        data_points.append([0,[totals[0]/count, totals[1]/count, totals[2]/count, totals[3]/count]])
    
    trackers = existing_trackers
    # Add new trackers 
    for i in range(len(bboxes)):
        if i in already_tracked:
            continue
        bbox = tuple(bboxes[i])
        tracker = cv2.TrackerMOSSE_create()
        tracker.init(frame, bbox)
        #cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+ bbox[2],bbox[1]+bbox[3]),(0, 0, 255), 2, 1)
        prevPos[tracker] = [bbox, obj_ids[i].item()]
        trackers.append(tracker)
        frame_bboxes.append((bbox, obj_ids[i]))

    curr_frame = {"bboxes": frame_bboxes, "frame": frame}
    frame_details[frame_id] = curr_frame

    if frame_id >= 2:
        disrupted_value = util.calcDisruptionValue(frame_details[frame_id-2], frame_details[frame_id-1], curr_frame)
        data_points[frame_id-1][0] = disrupted_value
        del frame_details[frame_id-2]
    frame_id += 1

    # cv2.imshow("frame", frame)
    # key = cv2.waitKey(0)
    # if key == 27:
    #     break

plot_result(data_points)
build_regression_model(data_points)

        

