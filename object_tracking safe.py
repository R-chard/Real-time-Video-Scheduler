import cv2
import sys
import os
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)

from object_detection import ObjectDetection
from multiprocessing.managers import BaseManager
from scheduler import Scheduler
from shared_frames import SharedFrames
import util
import config

class ObjectTracker():

    def __init__(self, videoDesc, hasDisplay):

        videoCount = len(videoDesc)
        # stream_files = list of files
        # objs_of_interest = list of objects of interest for each video stream
        stream_files, self.objs_of_interest = map(list,zip(*videoDesc))

        self.hasDisplay = hasDisplay

        # Initialize Object Detection
        self.od = ObjectDetection()

        # Init shared frames
        BaseManager.register('SharedFrames', SharedFrames)
        manager = BaseManager()
        manager.start()
        self.frames = manager.SharedFrames(videoCount)

        # Init scheduler
        self.scheduler = Scheduler(stream_files,self.frames)

        # Maps tracker to (previous obj coordinates, hasMissedPrev, class )
        self.prevPos = {}
        # self.trackers[i][j] = jth tracker for ith stream
        # Trackers are removed after missing 2 in a row
        self.trackers = [[] for _ in range(videoCount)]

        # frames_appeared[i] = all classes so far: how many times it appeared for frame i
        self.frames_appeared = [{} for _ in range(videoCount)]

        # TODO: Check if should move to groundTruth
        #self.prev_entities[i][j] = {obj_id: count} for each obj in stream i frame j
        self.prev_entities = [{} for _ in range(videoCount)]

        # obj_occurences[i] = {obj_id: [last appeared frame, freq, frames_left before it appears again ]} for stream i
        self.obj_occurences = [{} for _ in range(videoCount)] 

        # Used in the calculation of disruption
        if not config.REAL_TIME:
            self.sampling_rate = 1
            # { frame id: { Details: [(trackers, previously predicted pos)..] of each tracker
            #               Frame: frame }
            self.frame_details = {} 
            # self.values = (content, change, disruption)
            self.values = []
            # extract file name
            self.file_name = stream_files[0][stream_files[0].rindex("/")+1:stream_files[0].rindex(".")]

    def cleanUp(self):
        for processes in self.scheduler.streams:
            processes.join()
            processes.close()
        if self.hasDisplay:
            cv2.destroyAllWindows()

    def record_result(self, frame, stream_id, frame_id, found):
        if not config.REAL_TIME:
            filepath = os.path.join("res",self.file_name)
            content_file, change_file, disruption_file = filepath+"_content_value.csv", filepath+"_change_value.csv", filepath+"_disrupted_value.csv"
            content_values, change_values, disrupted_values = zip(*self.values)
            #value = util.calcValue(content_value, disrupted_value)

            with open(content_file, "w") as f:
                for i in range(len(content_values)):
                    if i>0:
                        content_value = 0.5* content_values[i] + 0.5* content_values[i-1]
                    else:
                        content_value = content_values[0]

                    f.write(str(i) + ", " + str(content_value) + ',\n')

            with open(change_file, "w") as f:
                for i in range(len(change_values)):
                    f.write(str(i)  + ", " + str(change_values[i]) + ',\n')

            with open(disruption_file, "w") as f:
                for i in range(len(disrupted_values)):
                    if i>0:
                        disrupted_value = 0.75* disrupted_values[i] + 0.25* disrupted_values[i-1]
                    else:
                        disrupted_value = disrupted_values[0]
                    f.write(str(i)  + ", " + str(disrupted_value) + ',\n')

        if self.hasDisplay:
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

        print("Frame ID: " + str(frame_id))
        
        found = [] # list of (coordinate of obj, whether it was discovered this frame)
        
        total_speed = 0

        # Object detection
        obj_ids, _, bboxes = self.od.detect(frame)
        
        # ------ Run on all objects ------------

        # Update entities spotted
        present_entities = {}
        for obj_id in obj_ids:
            obj_id = obj_id.item()
            
            if obj_id not in self.objs_of_interest[stream_id]:
                continue
            if obj_id not in present_entities:
                present_entities[obj_id] = 0
            if obj_id not in self.frames_appeared[stream_id]:
                self.frames_appeared[stream_id][obj_id] = 0

            present_entities[obj_id] += 1
            self.frames_appeared[stream_id][obj_id] += 1

        # Needed to calculate disruption value
        if not config.REAL_TIME:
            frame_detail = []
            for bbox in bboxes:
                new_tracker = cv2.legacy.TrackerKCF_create()
                new_tracker.init(frame,bbox)
                frame_detail.append((new_tracker, bbox))

        # ------ Run on tracked objects ------------

        # Check all bounding boxes discovered by existing trackers
        existing_trackers = []
        
        for i in range(len(self.trackers[stream_id])):
            tracker = self.trackers[stream_id][i]
            _, tracked_bbox = tracker.update(frame)

            # Revert to previous pos if tracked box failed
            if not tracked_bbox[2] and not tracked_bbox[3]:
                tracked_bbox = self.prevPos[tracker][0]

            found_by_dnn = False
            # Update tracker if DNN returns a slightly different location
            j = 0
            for bbox in bboxes:
                bbox = bbox.tolist()            
                if util.ifCentreIntersect(bbox, tracked_bbox) and self.prevPos[tracker][2] == obj_ids[j]:
                    tracker.init(frame,bbox)
                    # Update data based on new tracked locations
                    #speed = util.calcFrameLeft(bbox, self.prevPos[tracker][0], frame_diff)
                    #total_speed += speed
                    self.prevPos[tracker] = [bbox, False, obj_ids[j]]
                    found_by_dnn = True
                    break
                j += 1

            # obj detected
            if found_by_dnn:
                bboxes = np.delete(bboxes,j,0)
                obj_ids = np.delete(obj_ids,j,0)
                found.append((tracked_bbox,False))
                existing_trackers.append(tracker)

            # object persisting
            elif not self.prevPos[tracker][1]:
                found.append((tracked_bbox,False))
                existing_trackers.append(tracker)
                # TODO: consider adding to present objects

            else:
                # Remove tracker if dnn has missed more than 2x consecutively
                del self.prevPos[tracker]
        self.trackers[stream_id] = existing_trackers
        
        # ------ Run on objects that are newly detected ------------
        new_entities = {} # obj_ids: number of discovered objects in this frame

        for i in range(len(bboxes)):
            # Create trackers for it
            found.append((bboxes[i], True))
            tracker = cv2.legacy.TrackerKCF_create()
            tracker.init(frame,bboxes[i])
            self.prevPos[tracker] = [bboxes[i], False, obj_ids[i]]
            self.trackers[stream_id].append(tracker)

            # Update entities spotted
            # TODO: if we should move next line higher up
            if obj_ids[i] not in self.objs_of_interest[stream_id]:
                continue
            if obj_ids[i] not in new_entities:
                new_entities[obj_ids[i]] = 0
            new_entities[obj_ids[i]] += 1

            # Update frame details for disruption calculations
            if not config.REAL_TIME:
                new_tracker = cv2.legacy.TrackerKCF_create()
                new_tracker.init(frame,bboxes[i])
                frame_detail.append((new_tracker, bboxes[i]))

        # Update prev_entities and remove outdated frames
        for k in self.prev_entities[stream_id].keys():
            if k <= frame_id - 5:
                del self.prev_entities[stream_id][k]
        self.prev_entities[stream_id][frame_id] = new_entities
        
        # ------ Calculate priority/ values after processing ------------

        # Only calculate in realtime
        if config.REAL_TIME:
            for obj_id,count in new_entities.items():
                # First time object appearing
                if obj_id not in self.obj_occurences[stream_id]:
                    self.obj_occurences[stream_id][obj_id] = [frame_id,count/(frame_id+1)]
                else:
                    prev_appearance, prev_rate = self.obj_occurences[stream_id][obj_id]
                    delta = 0.5
                    new_rate = count/(frame_id - prev_appearance)
                    mva_rate = new_rate * delta + prev-rate * (1-delta) 
                    self.obj_occurences[stream_id][obj_id] = [frame_id,mva_rate]
            # estimate_change_value()
            # estimate_priority_value()
            # estimate_disruption_value()
            
        # Only calculate in offline mode
        else:

            # total_new = sum(self.prev_entities.values()) * 0.5 + sum(new_entities.values()) * 0.5

            # Calculate content value
            content_value = util.calcContentValue(present_entities, self.frames_appeared[stream_id], frame_id )

            # Calculate change value
            change_value = util.calcChangeValue(self.prev_entities[stream_id], present_entities)
            self.values.append([content_value,change_value,0])

            # Calculate disruption value
            curr_frame = {"details": frame_detail, "frame": frame}
            self.frame_details[frame_id] = curr_frame
            past_frame_id = frame_id - 2*self.sampling_rate

            # Only store value if we are past the first few frames
            if past_frame_id >= 0:
                disrupted_frame_id = frame_id - self.sampling_rate
                disrupted_frame_details = self.frame_details[disrupted_frame_id]
                curr_frame_details = self.frame_details[frame_id]
                disrupted_value = util.calcDisruptionValue(self.frame_details[past_frame_id], disrupted_frame_details, curr_frame_details)
                
                self.values[disrupted_frame_id][2] = disrupted_value

                print("For frame ID: " + str(frame_id))
                print("Content value: " + str(self.values[disrupted_frame_id][0]))
                print("Change value: " + str(self.values[disrupted_frame_id[1]]))
                print("Disruption value: " + str(disrupted_value))
                
                del self.frame_details[past_frame_id]

        return self.record_result(frame, stream_id, frame_id, found)
        