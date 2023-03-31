import numpy as np
import time

# Data structure accessible by both scheduler and stream processes
class SharedFrames(object):

    def __init__(self,vid_count):
        # self.frames[i] = frame, frameID for stream i
        self.frames = [[np.empty(0), -1] for _ in range(vid_count)]
        self.deadline = [0 for _ in range(vid_count)]
        self.stream_length = [0 for _ in range(vid_count)]
        self.frame_dim = [0 for _ in range(vid_count)]

    def poll(self, i):
        frame = self.frames[i]
        self.frames[i] = [np.empty(0), -1]
        return frame

    def get(self, i):
        frame = self.frames[i]
        return frame

    def get_deadline(self, i):
        return self.deadline[i]

    def set_stream_length(self, i, stream_length):
        self.stream_length[i] = stream_length

    def set_frame(self, i, frame, frameID, fps):
        self.frames[i] = [frame,frameID]
        self.deadline[i] = time.time() + 1/fps

    def end_stream(self, i, fps):
        self.set_frame(i, np.empty(0), -2, fps)

    def get_total_stream_lengths(self):
        return sum(self.stream_length)
    
    def set_frame_dim(self, i, stream_width, stream_height):
        self.frame_dim[i] = [stream_width, stream_height]

    def get_frame_dim(self, i):
        return self.frame_dim[i]
    
    def isFrameEmpty(self, i):
        return not self.frames[i] or self.frames[i][0].size == 0
