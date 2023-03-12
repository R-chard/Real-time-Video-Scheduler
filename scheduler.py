import numpy as np
import time
from stream import Stream

class Scheduler():
    def __init__(self, stream_files, frames):

        self.frames = frames
        self.streams = [None for _ in range(len(stream_files))]
        # last return frame id 
        self.last_ret = [-1 for _ in range(len(stream_files))]
        # can be "edf", "prio" or "rr"
        # converts to "prio" after missing 2 in a row
        self.mode = "edf"
        # Stream of most recently returned frame
        self.most_rec_stream = -1

        # Priority of streams
        self.priority = [0 for _ in range(len(stream_files))]

        for i in range(len(stream_files)):
            self.streams[i] = Stream(i,stream_files[i],self.frames)
            self.streams[i].start()
        
        print("Scheduler connected to all streams")

    # Run through all streams, update priority
    def set_prio(self, stream_id, priority):
        self.priority[stream_id] = priority  
    
    # Return the next ( stream_id, frame, frameID, time elapsed since prev frame returned )
    def get_frame_data(self):
        best_stream_id = -1
        
        for i in range(self.frames.length()):
            if self.frames.isFrameEmpty(i):
                continue
            
            # Returns based on edf
            if self.mode == "edf":
                # If we missed more than one frame for this stream, we have missed a deadline
                if self.frames.get(i)[1] > self.last_ret[i] + 1:
                    print("Missed deadline on stream: " + str(self.frames.get(i)[1]))
                    self.mode = "prio"
                if best_stream_id == -1 or self.frames.get_deadline(i) < self.frames.get_deadline(best_stream_id):
                    best_stream_id = i

            # Returns based on highest priority 
            else:
                if best_stream_id == -1 or self.priority[i] > self.priority[best_stream_id]:
                    best_stream_id = i
            
        # if no frames can be found
        if best_stream_id == -1:
            all_streams_end = True
            for i in range(self.frames.length()):
                if self.frames.get(i)[1] != -2:
                    all_streams_end = False
                    break
            
            # Signals to object_tracker to terminate appication
            if all_streams_end:
                return [-1, np.empty(0), -2, 0]
            return [-1,np.empty(0),-1,0]

        best_frame = self.frames.poll(best_stream_id)
        ret = (best_stream_id, best_frame[0], best_frame[1], best_frame[1] - self.last_ret[best_stream_id])
        
        # Update internal data structures
        self.last_ret[best_stream_id] = best_frame[1]
        self.most_rec_stream = best_stream_id
        return ret
