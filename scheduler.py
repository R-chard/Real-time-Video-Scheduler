import numpy as np
from stream import Stream

STREAM_DIR = "media"

class Scheduler():
    def __init__(self, stream_files, frames):

        self.frames = frames
        self.streams = []
        for i in range(len(stream_files)):
            stream = Stream(i,stream_files[i],self.frames)
            self.streams.append(stream)
            stream.start()
        
        print("Scheduler connected to all streams")
    
    # returns stream_id and frame with the highest priority
    def get_frame(self):
        best_stream_id = -1

        for i in range(self.frames.length()):
            if self.frames.get(i)[0].size == 0:
                continue
            if best_stream_id == -1 or self.frames.get(i)[1] > self.frames.get(best_stream_id)[1]:
                best_stream_id = i
        
        if best_stream_id == -1:
            return [-1,np.empty(0)]

        ret = (best_stream_id,self.frames.get(best_stream_id)[0])
        self.frames.set_frame(best_stream_id, np.empty(0))
        return ret
    
    def set_priority(self,stream_id,new_prio):
        self.frames.set_prio(stream_id,new_prio)
