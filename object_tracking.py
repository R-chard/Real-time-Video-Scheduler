import cv2
import sys
import os
import numpy as np

from object_detection import ObjectDetection
from multiprocessing.managers import BaseManager
from scheduler import Scheduler
from shared_frames import SharedFrames
import util
import config

CHANGE_THRESHOLD = 6

class ObjectTracker():

    def __init__(self, videoDesc):

        videoCount = len(videoDesc)
        # stream_files = list of files
        # objs_of_interest = list of objects of interest for each video stream
        stream_files, self.objs_of_interest = map(list,zip(*videoDesc))

        # Initialize Object Detection
        self.od = ObjectDetection()

        # Init shared frames
        BaseManager.register('SharedFrames', SharedFrames)
        manager = BaseManager()
        manager.start()
        self.frames = manager.SharedFrames(videoCount)

        # Maps tracker to (previous obj coordinates, hasMissedPrev, class )
        self.prevPos = {}

        # self.trackers[i][j] = jth tracker for ith stream
        # Trackers are removed after missing 2 in a row
        self.trackers = [[] for _ in range(videoCount)]

        # frames_appeared[i] = all classes so far: how many times it appeared for frame i
        self.frames_appeared = [{} for _ in range(videoCount)]

        # self.values = (content, change, disruption)
        self.values = []

        # extract file name
        self.file_name = stream_files[0][stream_files[0].rindex("/")+1:stream_files[0].rindex(".")]

        # Used in the calculation of disruption
        if config.REAL_TIME:
            
            self.max_frames_processed = 0
            self.frames_processed = [0 for _ in range(videoCount)]
        else:
            self.sampling_rate = 1
            # { frame id: { bboxes: [], Frame: frame }
            self.frame_details = {} 

        # Init scheduler
        self.scheduler = Scheduler(stream_files,self.frames)
            
    def cleanUp(self):
        for processes in self.scheduler.streams:
            processes.join()
            processes.close()
        if config.HAS_DISPLAY:
            cv2.destroyAllWindows()

    def show_res(self, frame, frame_id, found):
        for bbox,is_newly_disc in found:   
            x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if is_newly_disc:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255), 2, 1)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0), 2, 1)
            cv2.putText(frame, "Frame: " + str(frame_id),(50,50),0,1,(0,0,255), 2)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            return False
        return True

    # Processing a single frame
    def run(self):
        stream_id, frame, frame_id, frame_diff = self.scheduler.get_frame_data()
        # Stream is done
        if frame_id == -2:
            return False
        # Waiting on stream to update
        if frame.size == 0:
            return True

        if config.REAL_TIME:
            print("Frame ID: " + str(frame_id))
        
        # obj_details[i] = [[Bbox, speed, Confidence, class]]
        obj_details = []
        
        if config.HAS_DISPLAY:
            found = [] # list of (coordinate of obj, whether it was discovered this frame)

        # Object detection
        obj_ids, confidences, bboxes = self.od.detect(frame)

        # ------ Run on previously tracked objects ------------
        # Check all bounding boxes discovered by existing trackers
        existing_trackers = []
        present_entities = {}
        tracked_bbox = []
        to_ignore = set()
        counted_obj_id = set()

        for i in range(len(self.trackers[stream_id])):
            tracker = self.trackers[stream_id][i]
            success, tracked_bbox = tracker.update(frame)
            # Revert to previous pos if tracked box failed
            if not success or not tracked_bbox[2] and not tracked_bbox[3]:
                tracked_bbox = self.prevPos[tracker][0]
                x,y,w,h = tracked_bbox
                if config.HAS_DISPLAY:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(150, 150, 120), 2, 1)

            found_by_dnn = False
            # Update tracker if DNN returns a slightly different location
            for j in range(len(bboxes)):

                # Ignore background objects
                obj_id = obj_ids[j].item()
                if obj_id not in self.objs_of_interest[stream_id] or j in to_ignore:
                    continue
                
                to_ignore.add(j)
                bbox = tuple(bboxes[j])         
                if util.ifCentreIntersect(bbox, tracked_bbox) and self.prevPos[tracker][2] == obj_id:
                    tracker.init(frame,bbox)

                    # Update data based on new tracked locations
                    speed = util.calcSpeed(bbox, self.prevPos[tracker][0], frame_diff)
                    obj_details.append([bbox,speed,confidences[j].item(),obj_id])
                    self.prevPos[tracker] = [bbox, False, obj_id]
                    found_by_dnn = True
                    break

            # object detected or persisted
            if found_by_dnn or not self.prevPos[tracker][1]:
                if config.HAS_DISPLAY:
                    found.append((tracked_bbox,False))

                existing_trackers.append(tracker)
                bbox,_,obj_id = self.prevPos[tracker]
                bbox = tuple(bbox)   

                # Add to present entities
                if obj_id not in present_entities:
                    present_entities[obj_id] = 0
                present_entities[obj_id] += 1

                # count how many times this object appeared
                if obj_id not in counted_obj_id:
                    counted_obj_id.add(obj_id) 
                    if obj_id not in self.frames_appeared[stream_id]:
                        self.frames_appeared[stream_id][obj_id] = 0
                    self.frames_appeared[stream_id][obj_id] += 1

                if not found_by_dnn:
                    self.prevPos[tracker][1] = True
            else:
                # Remove tracker if dnn has missed more than 2x consecutively
                del self.prevPos[tracker]
        self.trackers[stream_id] = existing_trackers

        # ------ Run on objects that are newly detected ------------
        new_entities = {} # obj_ids: number of discovered objects in this frame
        for i in range(len(bboxes)):
            obj_id = obj_ids[i].item()
            # Ignore background objects
            if obj_id not in self.objs_of_interest[stream_id] or i in to_ignore:
                continue

            if config.HAS_DISPLAY:
                found.append((bboxes[i], True))

            # Create trackers for new object
            tracker = cv2.TrackerKCF_create()
            bbox = tuple(bboxes[i])
            tracker.init(frame,bbox)
            self.prevPos[tracker] = [bbox, False, obj_id]
            self.trackers[stream_id].append(tracker)

            # Update entities spotted
            if obj_id not in new_entities:
                new_entities[obj_id] = 0
            if obj_id not in present_entities:
                present_entities[obj_id] = 0
            if obj_id not in self.frames_appeared[stream_id]:
                self.frames_appeared[stream_id][obj_id] = 0
            new_entities[obj_id] += 1
            present_entities[obj_id] += 1
            self.frames_appeared[stream_id][obj_id] += 1

            # Indicates new obj
            obj_details.append((bbox,None,confidences[i].item(),obj_id))
        
        # ------ Calculate priority/ values after processing ------------

        # Only calculate in offline mode
        if not config.REAL_TIME:

            # Calculate content value
            content_value = util.calcContentValue(present_entities, self.frames_appeared[stream_id], frame_id+1, frame_id+1)

            # Calculate change value
            change_value = util.calcChangeValue(new_entities, present_entities, obj_details)
            self.values.append([content_value,change_value,0])

            # Calculate disruption value
            curr_frame = {"bboxes": [(dets[0],dets[3]) for dets in obj_details], "frame": frame}
            self.frame_details[frame_id] = curr_frame
            past_frame_id = frame_id - 2*self.sampling_rate

            # Only store value if we are past the first few frames
            if past_frame_id >= 0:
                disrupted_frame_id = frame_id - self.sampling_rate
                curr_frame_details = self.frame_details[frame_id]
                disrupted_value = util.calcDisruptionValue(self.frame_details[past_frame_id], self.frame_details[disrupted_frame_id], curr_frame_details)
                
                self.values[disrupted_frame_id][2] = disrupted_value

                print("For frame ID: " + str(disrupted_frame_id))
                # print("Content value: " + str(self.values[disrupted_frame_id][0]))
                # print("Change value: " + str(self.values[disrupted_frame_id][1]))
                print("Disruption value: " + str(disrupted_value))
                
                del self.frame_details[past_frame_id]
    
        self.scheduler.update(stream_id, frame_id, new_entities, obj_details)

        if config.HAS_DISPLAY:
            return self.show_res(frame, frame_id, found)
        else:
            return True

         
        