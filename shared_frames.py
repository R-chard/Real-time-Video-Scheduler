import numpy as np

class SharedFrames(object):

    def __init__(self,stream_dir):
        self.frames = [[np.empty(0),0] for _ in range(stream_dir)]
    
    def length (self):
        return len(self.frames)

    def get(self, i):
        return self.frames[i]

    def set_prio(self, i, prio):
        self.frames[i][1] = prio

    def set_frame(self, i, frame):
        self.frames[i][0] = frame