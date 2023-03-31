import math
import cv2
import config
import numpy as np
from shapely import box, union_all

# A single file containing the calculations used in this dissertation

# Calculate IOU between 2 bounding boxes
def calcIOU(box1, box2):
    # no overlap
    if not box2[0] <= box1[0]+box1[2] <= box2[0] + box2[2] and not box1[0] <= box2[0]+box2[2] <= box1[0] + box1[2]:
        return 0
    if not box2[1] <= box1[1]+box1[3] <= box2[1] + box2[3] and not box1[1] <= box2[1]+box2[3] <= box1[1] + box1[3]:
        return 0
    intersect_w = min(box1[0]+box1[2],box2[0]+box2[2]) - max(box1[0],box2[0])
    intersect_h = min(box1[1]+box1[3],box2[1]+box2[3]) - max(box1[1],box2[1])
    intersect_area = intersect_w * intersect_h
    union_area = box1[2]*box1[3] + box2[2]*box2[3] - intersect_area

    return intersect_area/union_area

# Predict which objects are present in the frame
def predict_present_entities(obj_details, frame_diff, stream_width, stream_height):
    if not obj_details:
        return {}
    est_present = {}
    avg_x_speed = sum(obj[2][0] for obj in obj_details.values() if obj[2]) / len(obj_details)
    avg_y_speed = sum(obj[2][1] for obj in obj_details.values() if obj[2]) / len(obj_details)

    for bbox, obj_id, speed,_ in obj_details.values():
        if not speed:
            speed = [avg_x_speed,avg_y_speed]

        dist_x, dist_y = speed[0] * frame_diff, speed[1] * frame_diff 

        left_bound = bbox[0] + dist_x
        right_bound = bbox[0] + bbox[2] + dist_x
        top_bound = bbox[1] + bbox[3] + dist_y
        bot_bound = bbox[1] + dist_y

        # verify that objects are still in frame
        if left_bound >= 0 and right_bound <= stream_width and top_bound <= stream_height and bot_bound >= 0:
            if obj_id not in est_present:
                est_present[obj_id] = 0
            est_present[obj_id] += 1
    return est_present

# Used real time to predict the dynamism of a frame
def estimate_dynamism_value(prev_new_entities, obj_details, frame_diff, stream_width, stream_height):
    predicted_present = predict_present_entities(obj_details, frame_diff,stream_width, stream_height)
    
    avg_new_entity = sum(prev_new_entities)/len(prev_new_entities)
    # -1 is a filler class since obj class does not matter in dynamism
    predict_new = {-1:avg_new_entity}
    predicted_present.update(predict_new)
    return calcDynamismValue(predict_new, predicted_present, obj_details, stream_width, stream_height)

# Used real time to predict the impact of a frame using the linear regression model
def estimate_impact_value(model, obj_details):
    if not obj_details:
        return 0

    total_count = 0
    impact = 0
    for obj in obj_details.values():
        bbox, _, speed, _ = obj
        if not speed:
            continue
        mod_speed = (speed[0]**2 + speed[1]**2)**0.5
        predict = model.predict(np.array([[mod_speed]]))[0]
        if 0<predict<1:
            impact += (predict - predict**2)**0.5
        total_count += 1
    if total_count == 0:
        return 0
    return impact/total_count
    
# Used real time to predict the content value of a frame
def estimate_content_value(obj_details, frame_diff, frames_appeared, frames_processed, max_frames_processed, stream_width, stream_height):
    estimated_present_entities = predict_present_entities(obj_details, frame_diff, stream_width, stream_height)
    expected_appeared = {}
    for k in frames_appeared.keys():
        expected_appeared[k] = frames_appeared[k] + 1
    return calcContentValue(estimated_present_entities, expected_appeared, frames_processed, max_frames_processed)

# Estimate speed of object relative to its trajectory
def conv_relative_speed(speed_x, speed_y, bbox, stream_width, stream_height):

    centre_x = bbox[0] + bbox[2]/2 
    centre_y = bbox[1] + bbox[3]/2 
    factor = 0.5

    if not speed_x and not speed_y:
        return 0
    elif not speed_x:
        return abs(speed_y/stream_height)**factor
    elif not speed_y:
        return abs(speed_x/stream_width)**factor
    else:
        gradient = speed_y/speed_x
        c = centre_y - gradient * centre_x
        speed = (speed_x**2 + speed_y**2)**0.5
        
        projected_min_x = 0
        projected_min_y = 0 * gradient + c
        
        projected_max_x = stream_width
        projected_max_y = stream_width * gradient + c
        
        if gradient > 0:
            if projected_min_y < 0:
                projected_min_y = 0
                projected_min_x = (0 - c)/gradient

            if projected_max_y > stream_height:
                projected_max_y = stream_height
                projected_max_x = (stream_height - c)/gradient
            
        else:
            if projected_min_y > stream_height:
                projected_min_y = stream_height
                projected_min_x = (stream_height - c)/gradient
            
            if projected_max_y < 0:
                projected_max_y = 0
                projected_min_x = (0 - c)/gradient
        
        if projected_max_y == projected_min_y:
            if projected_max_x == projected_min_x:
                return 0
            return abs(speed_x/ (projected_max_x - projected_min_x)) **factor
        return abs(speed_y/ (projected_max_y - projected_min_y)) **factor

# Calculates how much of an object is occluded
def percent_occluded(box1, all_boxes):
    iou_percent = 0
    all_occluded = []
    size = box1[2] * box1[3]

    for box2 in all_boxes:
        # Same box
        if box2[0] == box1[0] and box2[1] == box1[1] and box2[2] == box1[2] and box2[3] == box1[3]:
            continue

        # No overlap
        if not box2[0] <= box1[0]+box1[2] <= box2[0] + box2[2] and not box1[0] <= box2[0]+box2[2] <= box1[0] + box1[2]:
            continue
        if not box2[1] <= box1[1]+box1[3] <= box2[1] + box2[3] and not box1[1] <= box2[1]+box2[3] <= box1[1] + box1[3]:
            continue
        
        # x1,y1,x2,y2 of intersected region
        occluded = (max(box1[0],box2[0]),max(box1[1],box2[1]),min(box1[0]+box1[2],box2[0]+box2[2]),min(box1[1]+box1[3],box2[1]+box2[3]))
        all_occluded.append(box(*occluded))

    return union_all(all_occluded).area/size

# Calculates if the centre of an object is within another    
def ifCentreIntersect(box1, box2):
    centre_box1_x = box1[0] + (box1[2] / 2)
    centre_box1_y = box1[1] + (box1[3] / 2)

    centre_box2_x = box2[0] + (box2[2] / 2)
    centre_box2_y = box2[1] + (box2[3] / 2)

    return (box1[0] <= centre_box2_x <= box1[0] + box1[2] and box1[1] <= centre_box2_y <= box1[1]+box1[3]) or (box2[0] <= centre_box1_x <= box2[0] + box2[2] and box2[1] <= centre_box1_y <= box2[1] + box2[3])

# Return speed of object ( split into [x_speed, y_speed] ) in pixels/frame 
def calcSpeed(prev_box, curr_box, frame_diff):
    centre_prev_x = prev_box[0] + (prev_box[2] / 2)
    centre_prev_y = prev_box[1] + (prev_box[3] / 2)

    centre_x = curr_box[0] + (curr_box[2] / 2)
    centre_y = curr_box[1] + (curr_box[3] / 2)

    speed_x = (centre_x - centre_prev_x) / frame_diff
    speed_y = (centre_y - centre_prev_y) / frame_diff 

    return (speed_x, speed_y)

# Calcuate content value of a frame
def calcContentValue(present_entities, frames_appeared, frames_processed, max_frames_processed):

    value = 0
    if not max_frames_processed:
        return 0

    for entity in present_entities.keys():
        value = max(value, (frames_processed/ (frames_appeared[entity] * max_frames_processed)))
    
    return value

# Calcuate dynamism value of a frame
def calcDynamismValue(new_entities, present_entities, obj_details, stream_width, stream_height):
    
    total_present = sum(present_entities.values())
    total_new = sum(new_entities.values())

    if total_present == 0:
        dynamism = 0
    else:
        dynamism = total_new/total_present
    
    count = 0
    if not obj_details:
        avg_relative_speed = 0
    else:
        total_speed = 0
        for bbox,_,speed,_ in obj_details.values():
            if not speed:
                continue
            total_speed += conv_relative_speed(speed[0], speed[1],bbox,stream_width, stream_height)
            count += 1
        avg_relative_speed = total_speed/count if count >1 else 0
    return dynamism * 0.2 + avg_relative_speed * 0.8

# Calcuate impact value of a frame
def calcImpactValue(prev_frame_details, disrupted_frame_details, curr_frame_details):
    impact = 0 
    total_count = 0
    
    for prev_bbox,obj_id in prev_frame_details["bboxes"]:
        included_tracker = cv2.TrackerMOSSE_create()
        included_tracker.init(prev_frame_details["frame"], prev_bbox)
        ok, predicted_bbox_dis = included_tracker.update(disrupted_frame_details["frame"])
        if not ok:
            continue

        # Find most likely entity in 2nd frame to update tracker 
        max_iou = 0
        likely_bbox_dis = None

        for actual_bbox_dis, d_obj_id in disrupted_frame_details["bboxes"]:
            if obj_id != d_obj_id:
                continue
            iou = calcIOU(actual_bbox_dis, predicted_bbox_dis)
            if iou> max_iou:
                likely_bbox_dis = actual_bbox_dis
                max_iou = iou

        if not likely_bbox_dis:
            continue
        included_tracker.init(disrupted_frame_details["frame"], likely_bbox_dis)

        # Create and update excluded tracker
        excluded_tracker = cv2.TrackerMOSSE_create()
        excluded_tracker.init(prev_frame_details["frame"], prev_bbox)
        _, included_bbox = included_tracker.update(curr_frame_details["frame"])
        _, excluded_bbox = excluded_tracker.update(curr_frame_details["frame"])

        includedIOU = 0
        excludedIOU = 0

        # includedIOU = not skipping frame i-1
        # excludedIOU = skipped frame i-1
        for actual_bbox,c_obj_id in curr_frame_details["bboxes"]:
            if c_obj_id != obj_id:
                continue
            includedIOU = max(includedIOU, calcIOU(actual_bbox, included_bbox))
            excludedIOU = max(excludedIOU, calcIOU(actual_bbox, excluded_bbox))

        # increase impact value if tracker causes object tracking to worsen
        diff = includedIOU - excludedIOU
        impact += (abs(diff)**0.5) * (-1 if diff<0 else 1)
        total_count += 1

    if total_count == 0:
        return 0
    return impact/total_count

def estimateValue(est_content_value, est_dynamism_value, est_impact_value):
    return calcValue(est_content_value, est_dynamism_value, est_impact_value)

def calcValue(content_value, dynamism_value, impact_value):
    return (content_value**0.5 + dynamism_value**0.5 + impact_value) /3

