import numpy as np
import time
from timeit import default_timer as timer
import config
import util
from stream import Stream

# Video analytics scheduler. Depending on the scheduling algorithm used, different data structures are created
class Scheduler():
    def __init__(self, stream_files, frames):

        self.frames = frames
        self.stream_files = stream_files
        videoCount = len(stream_files)
        self.streams = [None for _ in range(videoCount)]
        # last return frame id
        self.mode = config.SCHEDULER_MODE
        self.last_ret = [-1 for _ in range(videoCount)]
        self.predicted_values = []
        self.terminated_streams = set()
        self.all_return_first = False
        self.index = 0
        self.frames_processed = [0 for _ in range(videoCount)]
        self.jfi = 0

        if self.mode == "fv":    
            self.model = config.load_regression_model()
            #How many obj appeared in stream i from the last 5 processed frames
            self.prev_new_entities = [[0 for _ in range(5)] for _ in range(videoCount)]
            # Tracks how many times each object appeared
            self.frames_appeared = [{} for _ in range(videoCount)]
            # id: {last_pos, obj_id, speed,confidence}
            self.memory = [{} for _ in range(videoCount)]
            self.max_frames_processed = 0
            # used for enforcing rr for the first 2 frames
            self.fv_initial_frames = 2*videoCount
            self.fv_i = 0

            # used in determining window size
            self.window_size = 2*videoCount
            self.window_unprocessed = set()
            self.window_frames_left = self.window_size

            if config.FV_OPTIMISATION:
                self.priority = [0 for _ in range(videoCount)]

        elif self.mode == "rr":
            self.next_i = 0
        
        if config.REAL_TIME:
            # Used for evaluation
            self.correct_decisions = 0
            self.total_decisions = 0
            self.total_frames_processed = 0
            self.last_obj_count = [0 for _ in range(videoCount)]

     # Load frame values into memory for evaluation in real time
    def load_frame_vals(self, files):
        self.stored_vals = []
        
        for _file in files:
            stored_val = []
            file_paths = config.get_csv_paths(_file, False)
            for file_path in file_paths:
                with open(file_path,"r") as f:
                    file_data = []
                    for line in f.readlines():
                        if line == "":
                            continue
                        # ignore comma
                        file_data.append(float(line[:-1]))
                stored_val.append(file_data)
            self.stored_vals.append(stored_val)

    def get_active_streams_count(self):
        return len(self.stream_files) - len(self.terminated_streams)

    # Returns total number of objs present
    # Used in performance evaluation
    def get_total_objs(self):
        return sum(self.last_obj_count)

    def start_streams(self):
        for i in range(len(self.stream_files)):
            self.streams[i] = Stream(i,self.stream_files[i],self.frames)
            self.streams[i].start()

    # Calculate jains fairness index. 1/n worst case. 1 best case
    def calc_jfi(self):
        denom = sum(frames**2 for frames in self.frames_processed) * len(self.frames_processed)
        self.jfi = (sum(self.frames_processed)**2)/ denom

    def get_available_frames(self):
        available_frames = [None for _ in range(len(self.stream_files))]
        for i in range(len(self.stream_files)):
            if not self.frames.isFrameEmpty(i):
                available_frames[i] = self.frames.get(i)[1]
        
        return available_frames

    def get_total_frames(self):
        return self.frames.get_total_stream_lengths()
    
    def evaluate_decision(self, selected_stream):
        max_fv_id = -1
        max_prio = -2
        frame_id = -1
        choices = 0
        for i in range(len(self.stream_files)):
            frame_id = self.frames.get(i)[1]
            if frame_id <= 0:
                continue
            choices += 1
            if self.stored_vals[i][3][frame_id-1] > max_prio:
                max_prio = self.stored_vals[i][3][frame_id-1]
                max_fv_id = i
        
        # only count if there are choices to be make
        if choices > 1:
            if selected_stream == max_fv_id:
                self.correct_decisions += 1
            self.total_decisions += 1
        self.total_frames_processed += 1

    def get_decision_accuracy(self):
        if not self.total_decisions:
            print("No decisions made. Increase FPS")
            return 0.0
        return self.correct_decisions/self.total_decisions

    # set object details of scheduler
    # input obj_details is format of {id: bbox, obj_id, confidence}
    # Only called in fv mode
    def update(self, stream_id, frame_id, frame_diff, obj_details):
        # for evaluation
        if config.REAL_TIME:
            self.last_obj_count[stream_id] = len(obj_details)
        # for stateful scheduler
        if config.SCHEDULER_MODE == "fv":
            existing_objs = {}
            obj_types_present = set()
            for id,(bbox, obj_id, _ , _) in self.memory[stream_id].items():
                if id in obj_details:
                    speed = util.calcSpeed(obj_details[id][0],bbox,frame_diff)
                    # Removes unfound objects from memory
                    existing_objs[id] = [obj_details[id][0], obj_id, speed, obj_details[id][2]]
                    if obj_id not in obj_types_present:
                        obj_types_present.add(obj_id)
                    del obj_details[id]

            self.memory[stream_id] = existing_objs
            
            # new entities
            for id,(bbox, obj_id, confidence) in obj_details.items():
                # No speed for new objects
                self.memory[stream_id][id] = [bbox,obj_id,None,confidence]
                if obj_id not in obj_types_present:
                    obj_types_present.add(obj_id)
                
            new_entities = len(obj_details)
            self.prev_new_entities[stream_id] = self.prev_new_entities[stream_id][1:] + [new_entities]

            # Update amount of frames appeared for each obj type
            for obj_id in obj_types_present:
                if obj_id not in self.frames_appeared[stream_id]:
                    self.frames_appeared[stream_id][obj_id] = 0
                self.frames_appeared[stream_id][obj_id] += 1

            # optimisation
            if config.FV_OPTIMISATION:
                stream_height, stream_width = self.frames.get_frame_dim(stream_id)
                est_content_value = util.estimate_content_value(self.memory[stream_id], frame_diff, \
                    self.frames_appeared[stream_id], self.frames_processed[stream_id], self.max_frames_processed, \
                    stream_width, stream_height)
                est_dynamism_value = util.estimate_dynamism_value(self.prev_new_entities[stream_id], self.memory[stream_id], frame_diff, stream_width, stream_height)
                est_impact_value = util.estimate_impact_value(self.model, self.memory[stream_id])
                priority = util.estimateValue(est_content_value, est_dynamism_value, est_impact_value)
                self.priority[stream_id] = priority
        
    # priority algorithm
    def get_highest_prio(self):
        best_stream_id = -1
        best_priority = -1
        
        # Run through each frame twice initially
        if self.fv_initial_frames > 0:
            best_stream_id = self.fv_i
            if not self.frames.isFrameEmpty(self.fv_i):
                self.fv_i = (self.fv_i+1)% len(self.streams)
                self.fv_initial_frames -= 1
        else:
            for i in range(len(self.streams)):
                if self.frames.isFrameEmpty(i):
                    if self.frames.get(i)[1] == -2 and i not in self.terminated_streams:
                        if len(self.terminated_streams) == 0 and not self.jfi:
                            self.calc_jfi()
                        self.terminated_streams.add(i)
                        if config.REAL_TIME:
                            self.last_obj_count[i] = 0
                    if len(self.terminated_streams) == len(self.streams):
                        return -2
                    continue
                # Uncomment for impact of window on JFI
                # if False:
                #     pass
                if self.window_frames_left <= len(self.window_unprocessed):
                    best_stream_id = next(iter(self.window_unprocessed))
                else:
                    if config.FV_OPTIMISATION:
                        if self.priority[i] > best_priority:
                            best_stream_id = i
                            best_priority = self.priority[i]
                    else:
                        frame_id = self.frames.get(i)[1]
                        frame_diff = frame_id- self.last_ret[i]
                        stream_height, stream_width = self.frames.get_frame_dim(i)
                        est_content_value = util.estimate_content_value(self.memory[i], frame_diff, \
                            self.frames_appeared[i], self.frames_processed[i], self.max_frames_processed, \
                            stream_width, stream_height)
                        est_dynamism_value = util.estimate_dynamism_value(self.prev_new_entities[i], self.memory[i], frame_diff, stream_width, stream_height)
                        est_impact_value = util.estimate_impact_value(self.model, self.memory[i])
                        priority = util.estimateValue(est_content_value, est_dynamism_value, est_impact_value)
                        
                        self.predicted_values.append([est_content_value, est_dynamism_value, est_impact_value, priority])
                        
                        if priority> best_priority:
                            best_stream_id = i
                            best_priority = priority

            if best_stream_id in self.window_unprocessed:
                self.window_unprocessed.remove(best_stream_id)
            if len(self.window_unprocessed) == 0:
                self.window_unprocessed = set(s for s in range(len(self.streams)) if s not in self.terminated_streams)
                self.window_frames_left = 2*(len(self.streams) - len(self.terminated_streams))
        return best_stream_id

    def get_next(self):
        i = len(self.streams)

        while i > 0:
            if self.frames.isFrameEmpty(self.next_i):
                # Stream just ended
                if self.frames.get(self.next_i)[1] == -2 and self.next_i not in self.terminated_streams:
                    if len(self.terminated_streams) == 0 and not self.jfi:
                        self.calc_jfi()
                    self.terminated_streams.add(self.next_i)
                    if config.REAL_TIME:
                        self.last_obj_count[self.next_i] = 0
                    if len(self.terminated_streams) == len(self.streams):
                        return -2
                self.next_i = (self.next_i+1)%len(self.streams)
                i -= 1
                continue
            # Return first non empty frame
            stream_id = self.next_i
            self.next_i = (self.next_i+1)%len(self.streams)
            return stream_id
        return -1

    # edf
    def get_edf(self):
        earliest_deadline_i = -1
        for i in range(len(self.streams)):
            if self.frames.isFrameEmpty(i):
                if self.frames.get(i)[1] == -2 and i not in self.terminated_streams:
                    if len(self.terminated_streams) == 0 and not self.jfi:
                        self.calc_jfi() 
                    self.terminated_streams.add(i)
                    if config.REAL_TIME:
                        self.last_obj_count[i] = 0
                    if len(self.terminated_streams) == len(self.streams):
                        return -2
                continue

            elif earliest_deadline_i == -1 or self.frames.get_deadline(i) < self.frames.get_deadline(earliest_deadline_i):
                earliest_deadline_i = i

        return earliest_deadline_i

    # Return the next ( stream_id, frame, frameID, time elapsed since prev frame returned )
    def get_frame_data(self):
        if self.mode == "rr":
            i = self.get_next()
        elif self.mode == "fv":
            i = self.get_highest_prio()
        elif self.mode == "edf":
            i = self.get_edf()
        # No active streams
        if i == -1:
            return [-1, np.empty(0), -1, 0]
        # All streams ended
        elif i == -2:
            return [-1, np.empty(0), -2, 0]

        # Remove frame and update internal data structures
        if config.REAL_TIME:
            self.evaluate_decision(i)

        self.frames_processed[i] += 1
        if self.mode == "fv":
            self.max_frames_processed = max(self.max_frames_processed, self.frames_processed[i])

        best_frame = self.frames.poll(i)
        frame_diff = best_frame[1] - self.last_ret[i]
        ret = (i, best_frame[0], best_frame[1], frame_diff)
        self.last_ret[i] = best_frame[1]
        
        return ret
