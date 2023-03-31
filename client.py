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
import psutil
import time

# Simulate a client real-time video analytic application

class Client():

    def __init__(self, videoDesc):

        videoCount = len(videoDesc)
        # stream_files = list of files
        # objs_of_interest = list of objects of interest for each video stream
        self.stream_files, self.objs_of_interest = map(list,zip(*videoDesc))

        # Init shared frames
        BaseManager.register('SharedFrames', SharedFrames)
        manager = BaseManager()
        manager.start()
        self.frames = manager.SharedFrames(videoCount)

        # self.trackers[i] = {tracker: (previous obj coordinates, obj class, speed, assigned_id, confidence)} for the ith stream
        self.trackers = [{} for _ in range(videoCount)]

        # self.values = (content, dynamism, impact, overall)
        self.values = []
        self.last_assigned_id = [0 for _ in range(videoCount)]

        # Used in the calculation of impact
        if config.REAL_TIME:
            self.max_frames_processed = 0
            self.frames_processed = [0 for _ in range(videoCount)]
            self.choice_times = []
            self.application_times = []
            self.cpu_loads = []
            self.cpu_memory = []
            self.total_obj = []
            self.stream_count = []
        else:
            self.sampling_rate = 1
            # { frame id: { bboxes: [], Frame: frame }
            self.frame_details = {}
            # frames_appeared[i] = all classes so far: how many times it appeared for frame i
            self.frames_appeared = [{} for _ in range(videoCount)]        
            
    # cleanup function after program runs
    def cleanUp(self):
        for processes in self.scheduler.streams:
            processes.join()
            processes.close()
        if config.HAS_DISPLAY:
            cv2.destroyAllWindows()

    # show output on screen if there is display
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
    
    def start_scheduler(self):
        self.scheduler = Scheduler(self.stream_files,self.frames)
        if config.REAL_TIME:
            self.scheduler.load_frame_vals(self.stream_files)
        self.scheduler.start_streams()
        
        time.sleep(1)
        # Give time for all streams to be initiated initialize Object Detection
        self.od = ObjectDetection()
                
    def get_values(self):
        return (self.values,self.scheduler.predicted_values)
        
    # Processing a single frame
    def run(self):
        start = time.process_time()
        stream_id, frame, frame_id, frame_diff = self.scheduler.get_frame_data()
        # Simulation complete
        if frame_id == -2:
            return False
        if frame.size == 0:
            return True
        end = time.process_time()
        selection_time = end - start

        obj_details = {}
        
        if config.HAS_DISPLAY:
            found = []

        start2 = time.process_time()
        # Object detection
        obj_ids, confidences, bboxes = self.od.detect(frame)

        # ------ Run on previously tracked objects ------------
        # Check all bounding boxes discovered by existing trackers
        existing_trackers = {}
        present_entities = {}
        tracked_bbox = []
        to_ignore = set()
        counted_obj_id = set()

        for tracker,(prev_bbox, prev_obj_id, _, assigned_id, _) in self.trackers[stream_id].items():
            success, tracked_bbox = tracker.update(frame)
            # Revert to previous pos if tracked box failed
            if not success or not tracked_bbox[2] and not tracked_bbox[3]:
                tracked_bbox = prev_bbox
                x,y,w,h = tracked_bbox
                if config.HAS_DISPLAY:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(150, 150, 120), 2, 1)

            max_iou = 0
            max_j = -1

            # Update tracker if DNN returns a slightly different location
            for j in range(len(bboxes)):
                # Ignore background objects
                obj_id = obj_ids[j].item()
                if obj_id not in self.objs_of_interest[stream_id] or j in to_ignore or prev_obj_id != obj_id:
                    continue
                
                bbox = tuple(bboxes[j])
                iou = util.calcIOU(bbox, tracked_bbox)    
                if iou > max_iou:
                    max_j = j
                    max_iou = iou

            # Object detected
            if max_j != -1:

                # Update data based on new tracked locations
                bbox = tuple(bboxes[max_j])
                obj_id = obj_ids[max_j].item()
                confidence = confidences[max_j].item()
                speed = util.calcSpeed(prev_bbox, bbox, frame_diff)
                tracker.init(frame, bbox)
                existing_trackers[tracker] = [bbox, obj_id, speed, assigned_id,confidence]
                to_ignore.add(max_j)
                obj_details[assigned_id] = [bbox,obj_id,confidence]

                if config.HAS_DISPLAY:
                    found.append((bbox,False))

                if not config.REAL_TIME:
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

        self.trackers[stream_id] = existing_trackers

        # ------ Run on objects that are newly detected ------------
        new_entities = {}
        for i in range(len(bboxes)):
            obj_id = obj_ids[i].item()
            # Ignore background objects
            if obj_id not in self.objs_of_interest[stream_id] or i in to_ignore:
                continue

            if config.HAS_DISPLAY:
                found.append((bboxes[i], True))

            # Create trackers for new object
            tracker = cv2.TrackerMOSSE_create()
            bbox = tuple(bboxes[i])
            confidence = confidences[i].item()
            assigned_id = self.last_assigned_id[stream_id]
            tracker.init(frame,bbox)
            self.trackers[stream_id][tracker] = [bbox,obj_id,None,assigned_id,confidence]
            self.last_assigned_id[stream_id] += 1

            if not config.REAL_TIME:
                # Update entities spotted
                if obj_id not in new_entities:
                    new_entities[obj_id] = 0
                if obj_id not in present_entities:
                    present_entities[obj_id] = 0

                if obj_id not in counted_obj_id:
                    counted_obj_id.add(obj_id) 
                    if obj_id not in self.frames_appeared[stream_id]:
                        self.frames_appeared[stream_id][obj_id] = 0
                    self.frames_appeared[stream_id][obj_id] += 1
                new_entities[obj_id] += 1
                present_entities[obj_id] += 1
           
            obj_details[assigned_id] = [bbox,obj_id,confidence]
        
        # ------ Calculate priority/ values after processing ------------

        # Only calculate in offline mode
        if not config.REAL_TIME:
            stream_height, stream_width, _ = frame.shape
            # Calculate content value
            content_value = util.calcContentValue(present_entities, self.frames_appeared[stream_id], frame_id+1, frame_id+1)

            # Calculate dynamism value
            details = {assigned_id:[bbox,obj_id,speed,confidence] for (bbox, obj_id, speed, assigned_id, confidence) in self.trackers[stream_id].values()}
            dynamism_value = util.calcDynamismValue(new_entities, present_entities, details, stream_width, stream_height)
            self.values.append([content_value,dynamism_value,0,0])

            # Calculate impact value
            curr_frame = {"bboxes": [(tuple(bboxes[i]),obj_ids[i]) for i in range(len(bboxes))], "frame": frame}
            self.frame_details[frame_id] = curr_frame
            past_frame_id = frame_id - 2*self.sampling_rate

            # Only store value if we are past the first few frames
            if past_frame_id >= 0:
                disrupted_frame_id = frame_id - self.sampling_rate
                curr_frame_details = self.frame_details[frame_id]
                disrupted_value = util.calcImpactValue(self.frame_details[past_frame_id], self.frame_details[disrupted_frame_id], curr_frame_details)
                
                self.values[disrupted_frame_id][2] = disrupted_value
                total_value = util.calcValue(self.values[disrupted_frame_id][0],self.values[disrupted_frame_id][1],disrupted_value)
                self.values[disrupted_frame_id][3] = total_value

                print("For frame ID: " + str(disrupted_frame_id))
                print("Content value: " + str(self.values[disrupted_frame_id][0]))
                print("Dynamism value: " + str(self.values[disrupted_frame_id][1]))
                print("Impact value: " + str(disrupted_value))
                
                del self.frame_details[past_frame_id]
        if config.REAL_TIME and frame_id != 0:
            end2 = time.process_time()
            self.application_times.append(end2-start2)

            self.cpu_loads.append(psutil.cpu_percent())
            self.cpu_memory.append(psutil.virtual_memory().percent)
        
        start3 = time.process_time()
        self.scheduler.update(stream_id, frame_id, frame_diff, obj_details)
        end3 = time.process_time()

        if config.REAL_TIME and frame_id != 0:
            self.choice_times.append(end3 - start3 + selection_time)
            self.total_obj.append(self.scheduler.get_total_objs())
            self.stream_count.append(self.scheduler.get_active_streams_count())
        
        if config.HAS_DISPLAY:
            return self.show_res(frame, frame_id, found)
        else:
            return True

         
        