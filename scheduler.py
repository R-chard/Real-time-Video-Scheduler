import numpy as np
import time
from timeit import default_timer as timer
import config
import util
from stream import Stream

class Scheduler():
    def __init__(self, stream_files, frames):

        self.frames = frames
        videoCount = len(stream_files)
        self.streams = [None for _ in range(videoCount)]
        # last return frame id 
        self.last_ret = [-1 for _ in range(videoCount)]
        self.frames_processed = [0 for _ in range(videoCount)]
        self.mode = config.SCHEDULER_MODE
        self.model = config.load_regression_model()

        # Tracks content of tracked objects
        self.obj_details = [[] for _ in range(videoCount)]
        #self.prev_new_entities[i][j] = {obj_id: count} for each new obj in stream i frame j
        self.prev_new_entities = [{} for _ in range(videoCount)]

        # Tracks how many times each object appeared
        self.frames_appeared = [{} for _ in range(videoCount)]
        self.predicted_values = []
        self.frame_states = [{} for _ in range(videoCount)]
        self.next_i = 0
        self.max_frames_processed = 0

        for i in range(len(stream_files)):
            self.streams[i] = Stream(i,stream_files[i],self.frames)
            self.streams[i].start()
        
        print("Scheduler connected to all streams")
        
    # set object details of scheduler
    # obj_details is format of {id: bbox, obj_id}
    def update(self, stream_id, frame_id, new_entities, present_entities, obj_details):

        # Update new entities
        updated_prev_entities = {}
        for k,v in self.prev_new_entities[stream_id].items():
            if k > frame_id - 5:
                updated_prev_entities[k] = v
        updated_prev_entities[frame_id] = new_entities
        self.prev_new_entities[stream_id] = updated_prev_entities

        self.obj_details[stream_id] = obj_details

        # Update count on how many frames each object type appeared
        for obj_id in present_entities.keys():
            if obj_id not in self.frames_appeared[stream_id]:
                self.frames_appeared[stream_id][obj_id] = 0
            self.frames_appeared[stream_id][obj_id] += 1
        
    # Run through all streams, update priority
    def get_highest_prio(self):
        best_stream_id = -1
        best_priority = -1
        all_ended = True
        for i in range(len(self.streams)):
            if self.frames.isFrameEmpty(i):
                if all_ended and self.frames.get(i)[1] != -2:
                    all_ended = False
                continue
            
            all_ended = False
            frame_id = self.frames.get(i)[1]
            frame_diff = frame_id- self.last_ret[i]
            stream_height, stream_width, _ = self.frames.get(i)[0].shape
            est_content_value = util.estimate_content_value(self.obj_details[i], frame_diff, self.frames_appeared[i], self.frames_processed[i], self.max_frames_processed)
            est_change_value = util.estimate_change_value(self.prev_new_entities[i], self.obj_details[i], frame_diff, stream_width, stream_height)
            est_disruption_value = util.estimate_disruption_value(self.model, self.obj_details[i])

            priority = util.estimateValue(est_content_value, est_change_value, est_disruption_value)
            
            self.predicted_values.append([est_content_value, est_change_value, est_disruption_value, priority])
            
            if priority> best_priority:
                best_stream_id = i
                best_priority = priority
        
        if all_ended:
            return (-2,0)
        return (best_stream_id, best_priority)

    # round robin
    def get_next(self):
        i = len(self.streams)
        all_ended = True

        while i > 0:
            if self.frames.isFrameEmpty(self.next_i):
                if all_ended and self.frames.get(self.next_i)[1] != -2:
                    all_ended = False
                self.next_i = (self.next_i+1)%len(self.streams)
                i -= 1
                continue
            # Return first non empty frame
            return self.next_i

        if all_ended:
            return -2
        return -1

    # edf
    def get_edf(self):
        earliest_deadline_i = -1
        all_ended = True

        for i in range(len(self.streams)):
            if self.frames.isFrameEmpty(i):
                if all_ended and self.frames.get(i)[1] != -2:
                    all_ended = False
                continue
            all_ended = False
            if self.frames.get_deadline(i) < self.frames.get_deadline(earliest_deadline_i):
                earliest_deadline_i = i

        if all_ended:
            return -2
        return earliest_deadline_i

    # Return the next ( stream_id, frame, frameID, time elapsed since prev frame returned )
    def get_frame_data(self):
        i = -1
        priority = None
        if self.mode == "rr":
            i = self.get_next()
        elif self.mode == "prio":
            i, priority = self.get_highest_prio()
        elif self.mode == "edf":
            i = self.get_edf()
        # No active streams
        if i == -1:
            return [-1,np.empty(0),-1,0]
        # All streams ended
        elif i == -2:
            return [-1, np.empty(0), -2, 0]

        # Remove frame
        best_frame = self.frames.poll(i)
        ret = (i, best_frame[0], best_frame[1], best_frame[1] - self.last_ret[i])
        
        # Update internal data structures
        self.last_ret[i] = best_frame[1]
        self.frames_processed[i] += 1
        self.max_frames_processed = max(self.max_frames_processed, self.frames_processed[i])
        if config.REAL_TIME:
            print("Chose stream: " + str(i) + " with a priority of " + str(priority))
        return ret
