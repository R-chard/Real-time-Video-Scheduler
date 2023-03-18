from multiprocessing import Process
import cv2
import numpy as np
import time
import config
import os

# Represents a single video stream
class Stream(Process):
    def __init__(self,id,stream_file, shared_frames):
        super(Stream,self).__init__()
        self.id = id
        self.stream_file = stream_file
        self.shared_frames = shared_frames
        self.frame_no = 0

    def run(self):
        file_path = os.path.join(config.MEDIA_DIR, self.stream_file)
        capture = cv2.VideoCapture(file_path)
        self.fps = capture.get(cv2.CAP_PROP_FPS) / 15
        print("Initializing stream " + self.stream_file + " of length " + str(capture.get(cv2.CAP_PROP_FRAME_COUNT)) + " with a fps of " + str(self.fps))

        first_frame = True
        while capture.isOpened():

            success,frame = capture.read()
            if not success:
                break
            # Resize frame
            dim = config.get_resized_dim(frame)
            frame = cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)
            self.shared_frames.set_frame(self.id, frame, self.frame_no, self.fps)

            if config.REAL_TIME:
                time.sleep(1/self.fps)
            else:
                # Only process when done
                while not self.shared_frames.isFrameEmpty(self.id):
                    continue
            
            # Bug with first frame taking extra long for detector
            if first_frame:
                time.sleep(10/self.fps)
                first_frame = False
            self.frame_no += 1
            
        # Indicates that stream is done running
        self.shared_frames.set_frame(self.id, np.empty(0), -2, self.fps)

        print(self.stream_file + " stream ended")
        capture.release()