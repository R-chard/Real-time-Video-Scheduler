import math
import cv2
import config
from shapely import box, union_all

MAX_X = 960
MAX_Y = 540

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

# prev_entities[i]: { obj_ids of new_entities from curr_frame - i: count }
# prev_entities might not be consecutive anymore
# present_entities = { obj_ids of present_entities : count}
def estimate_change_value(prev_entities, estimated_present_entities):
    calcChangeValue(prev_entities, estimated_present_entities)
    
#def estimate_disruption_value():

# obj_details[i][0] = last known bbox of obj[i]
# obj_details[i][1] = last known speed of obj[i] (x /frame, y/ frame)
# obj_details[i][2] = class
# frame_diff = frame id of last frame processed
# max_frames_processed = maximum value of frame processed
# TODO: Consider freq of obj appearing to predict
def estimate_content_value(obj_details, frame_diff, frames_appeared, max_frames_processed):
    estimated_present_entities = {}
    
    for obj in obj_details:
        bbox, speed, obj_id = obj

        left_bound = bbox[0] + speed[0] * frame_diff
        right_bound = bbox[0] + bbox[2] + speed[0] * frame_diff
        top_bound = bbox[1] + bbox[3] + speed[1] * frame_diff
        bot_bound = bbox[1] + speed[1] * frame_diff

        # verify that objects are still in frame
        if left_bound > 0 and right_bound < config.SCREEN_MAX_WIDTH and top_bound < config.SCREEN_MAX_HEIGHT and bot_bound > 0:
            if obj_id not in estimated_present_entities:
                estimated_present_entities[obj_id] = 0
            estimated_present_entities[obj_id] += 1
    
    return calcContentValue(estimated_present_entities, frames_appeared, max_frames_processed)

def estimate_disruption_value()

# Calculates how many frames left before object moves out of frame
def calcFrameLeft(prevBox, newBox, frame_diff):
    centre_prev_x = prevBox[0] + (prevBox[2] / 2)
    centre_prev_y = prevBox[1] + (prevBox[3] / 2)

    centre_x = newBox[0] + (newBox[2] / 2)
    centre_y = newBox[1] + (newBox[3] / 2)

    if centre_prev_x == centre_x:
        gradient = float("inf")
        x_diff = 1
    else:
        gradient = max((centre_y - centre_prev_y)/ (centre_x - centre_prev_x),1)
        x_diff = centre_x - centre_prev_x
    c = centre_prev_y - gradient * centre_prev_x

    if gradient < 0:
        projected_x = 0
        projected_y = gradient * 0 + c

        if projected_y < 0:
            projected_y = 0
            projected_x = ( projected_y - c ) / gradient
        
    else:
        projected_x = MAX_X
        projected_y = gradient * MAX_X + c

        if projected_y > MAX_Y:
            projected_y = MAX_Y
            projected_x = ( projected_y - c ) / gradient
    
    multiplier = abs((projected_x - centre_x)/x_diff)

    return multiplier * frame_diff

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
    

def ifCentreIntersect(box1, box2):
    centre_box1_x = box1[0] + (box1[2] / 2)
    centre_box1_y = box1[1] + (box1[3] / 2)

    centre_box2_x = box2[0] + (box2[2] / 2)
    centre_box2_y = box2[1] + (box2[3] / 2)

    return box1[0] < centre_box2_x < box1[0] + box1[2] and box1[1] < centre_box2_y < box1[1]+box1[3] or box2[0] < centre_box1_x < box2[0] + box2[2] and box2[1] < centre_box1_y < box2[1] + box2[3]

# note that speed is returned in delta x/frame and delta y/frame 
def calcSpeed(prev_box, curr_box, frame_diff):
    centre_prev_x = prev_box[0] + (prev_box[2] / 2)
    centre_prev_y = prev_box[1] + (prev_box[3] / 2)

    centre_x = curr_box[0] + (curr_box[2] / 2)
    centre_y = curr_box[1] + (curr_box[3] / 2)

    speed_x = (centre_x - centre_prev_x) / frame_diff
    speed_y = (centre_y - centre_prev_y) / frame_diff 

    return (speed_x, speed_y)

# prev_entities[i]: { obj_ids of new_entities from curr_frame - i: count }
# present_entities = { obj_ids of present_entities : count}
# Adjust to include speed in calculations?
def calcChangeValue(prev_entities, present_entities):
    total_prev = 0
    for prev_entity in prev_entities.values():
        total_prev = sum(prev_entity.values())
    
    avg_entity = total_prev/ len(prev_entities)
    total_present = sum(present_entities.values())

    if total_present == 0:
        return 0
    return (avg_entity**0.5)/total_present

# present_entities = map of present entities : how many 
# frames_appeared = how many frames each entity appeared
# max_frames_processed = maximum number of frames processed for a single stream
def calcContentValue(present_entities, frames_appeared, frames_processed, max_frames_processed):

    value = 0
    total_present = sum(present_entities.values())

    for entity in present_entities.keys():
        value = max(value, (frames_processed/ (frames_appeared[entity] * max_frames_processed))**0.5)
    
    return value

# value ranges from -1 to 1
def calcDisruptionValue(prev_frame_details, disrupted_frame_details, curr_frame_details):
    disruption = 0 
    total_count = 0

    for p_tracker, p_bbox in prev_frame_details["details"]:
        p_tracker.update(disrupted_frame_details["frame"])
        success, included_bbox = p_tracker.update(curr_frame_details["frame"])

        includedIOU = 0

        if not success:
            continue

        # includedIOU = not skipping frame i-1
        # excludedIOU = skipped frame i-1
        for _, actual_bbox in curr_frame_details["details"]:
            includedIOU = max(includedIOU, calcIOU(actual_bbox,included_bbox))

        new_tracker = cv2.TrackerKCF_create()
        new_tracker.init(prev_frame_details["frame"], p_bbox)
        success_e, excludedIOU = new_tracker.update(curr_frame_details["frame"])

        if not success_e:
            excludedIOU = 0
        else:
            excludedIOU = calcIOU(actual_bbox, excludedIOU)

        # increase disruption value if tracker causes object tracking to worsen
        diff = includedIOU - excludedIOU
        if diff > 0:
            disruption += diff**2
        else:
            disruption -= diff**2
        total_count += 1

    if total_count == 0:
        return 0

    return disruption/total_count

def calcValue(content_value, change_value, disruption_value):
    return 2* content_value + change_value + disruption_value

