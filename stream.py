from multiprocessing import Process
import cv2
import numpy as np
import time
import config

# Represents a single video stream
class Stream(Process):
    def __init__(self,id,stream_file, shared_frames):
        super(Stream,self).__init__()
        self.id = id
        self.stream_file = stream_file
        self.shared_frames = shared_frames
        self.frame_no = 0

    def run(self):
        capture = cv2.VideoCapture(self.stream_file)
        #capture.set(cv2.CAP_PROP_POS_FRAMES, 46)
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        print("Initializing stream " + self.stream_file + " of length " + str(capture.get(cv2.CAP_PROP_FRAME_COUNT)) + " with a fps of " + str(self.fps))

        while capture.isOpened():
            if config.REAL_TIME:
                time.sleep(1/self.fps)
            else:
                # Only process when done
                if not self.shared_frames.isFrameEmpty(self.id):
                    continue 
            success,frame = capture.read()
            if not success:
                break
            # Resize frame
            dim = config.get_resized_dim(frame)
            frame = cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)
            self.shared_frames.set_frame(self.id, frame, self.frame_no, self.fps)
            self.frame_no += 1

        if config.REAL_TIME:
            time.sleep(1/self.fps)
        else:
            while not self.shared_frames.isFrameEmpty(self.id):
                continue
            
        # Indicates that stream is done running
        self.shared_frames.set_frame(self.id, np.empty(0), -2, self.fps)

        print("Stream ended")
        capture.release()