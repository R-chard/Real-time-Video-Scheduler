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
size_to_iou = []
speed_to_iou = []
confidence_to_iou = []
occlusion_percent_to_iou = []
prevPos = {}
trackers = []

od = ObjectDetection()
data_points = []
frame_id = 0

def plot_result(points, title, desc):
    plt.clf()
    y_points, x_points = zip(*points)

    x_points = np.array(x_points)
    y_points = np.array(y_points)
    gradient, c = np.polyfit(x_points,y_points,1)

    plt.ylabel("KCF prediction accuracy")
    plt.xlabel(desc)
    plt.scatter(x_points, y_points)
    
    print("Prediction results for: " + title)
    print("Gradient calculated to be: " + str(gradient))
    print("Y-Intercept calculated to be: " + str(c) + "\n")

    file_path = os.path.join(config.RES_DIR, title + ".png")
    plt.savefig(file_path)
    
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
    obj_ids, confidences, bboxes = od.detect(frame)

    already_tracked = set()
    existing_trackers = []

    print("Frame " + str(frame_id))

    for i in range(len(trackers)):
        _,tracked_bbox = trackers[i].update(frame)
        if not tracked_bbox[2] and not tracked_bbox[3]:
            del prevPos[trackers[i]]
            continue

        found = False
        for j in range(len(bboxes)):
            if j in already_tracked:
                continue

            # Renaming
            obj_id = obj_ids[j].item()
            confidence = round(confidences[j].item(),3)
            bbox = tuple((bboxes[j][0], bboxes[j][1], bboxes[j][2], bboxes[j][3]))

            # Found match
            if util.ifCentreIntersect(bbox, tracked_bbox) and prevPos[trackers[i]][1] ==  obj_id:
                iou = util.calcIOU(tracked_bbox, bbox)

                speed_x,speed_y = util.calcSpeed(prevPos[trackers[i]][0], bbox, 1)
                speed = round((speed_x**2 + speed_y**2)**0.5,1)
                size = bbox[2] * bbox[3]
                occlusion_percent = util.percent_occluded(bbox,bboxes)

                data_points.append([iou,[speed,size,confidence]])

                # Update results
                speed_to_iou.append((iou,speed))
                size_to_iou.append((iou,size))
                confidence_to_iou.append((iou,confidence))
                occlusion_percent_to_iou.append((iou,occlusion_percent))

                already_tracked.add(j)

                # update past position
                prevPos[trackers[i]][0] = bbox
                trackers[i].init(frame,bbox)
                found = True
                break
        if found:
            existing_trackers.append(trackers[i])
        else:
            del prevPos[trackers[i]]
    
    trackers = existing_trackers
    # Add new trackers
    for i in range(len(bboxes)):
        if i in already_tracked:
            continue
        bbox = tuple((bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]))
        new_tracker = cv2.TrackerKCF_create()
        new_tracker.init(frame, bbox)
        #cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+ bbox[2],bbox[1]+bbox[3]),(0, 0, 255), 2, 1)
        prevPos[new_tracker] = [bbox, obj_ids[i].item()]
        trackers.append(new_tracker)
    frame_id += 1

    # cv2.imshow("frame", frame)
    # key = cv2.waitKey(0)
    # if key == 27:
    #     break

plot_result(size_to_iou, "size", "Size of object")
plot_result(speed_to_iou, "speed", "Speed of object")
plot_result(confidence_to_iou, "confidence", "Confidence of DNN")
plot_result(occlusion_percent_to_iou, "occlusion", "Percentage of object occluded")
build_regression_model(data_points)

        

