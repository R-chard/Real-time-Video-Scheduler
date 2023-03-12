import numpy as np
import time

class SharedFrames(object):

    def __init__(self,stream_dir):
        # self.frames[i] = frame, frameID for stream i
        self.frames = [[np.empty(0), -1] for _ in range(stream_dir)]
        self.deadline = [0 for _ in range(stream_dir)]
    
    def length (self):
        return len(self.frames)

    def poll(self, i):
        frame = self.frames[i]
        self.frames[i] = [np.empty(0), -1]
        return frame

    def get(self, i):
        frame = self.frames[i]
        return frame

    def set_frame(self, i, frame, frameID, fps):
        self.frames[i] = [frame,frameID]
        self.deadline[i] = time.time() + 1/fps

    def get_deadlines(self, i):
        return self.deadline[i]
    
    def isFrameEmpty(self, i):
        return not self.frames[i] or self.frames[i][0].size == 0