from multiprocessing import Process
import cv2
import time
import config
import os

# Simulates a video stream
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
        self.fps = capture.get(cv2.CAP_PROP_FPS) * config.FPS_SCALE
        stream_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.shared_frames.set_stream_length(self.id, stream_length)
        stream_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        stream_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.shared_frames.set_frame_dim(self.id, stream_width, stream_height)

        print("Initializing stream " + self.stream_file + " of length " + str(stream_length) + " with a fps of " + str(self.fps))

        first_frame = True
        while capture.isOpened():

            success,frame = capture.read()
            if not success:
                break
            # Resize frame
            dim = config.get_resized_dim(frame)
            frame = cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)
            self.shared_frames.set_frame(self.id, frame, self.frame_no, self.fps)

            if not config.REAL_TIME or first_frame:
                first_frame = False
                # Bug with first frame taking extra long for detector to set up
                while not self.shared_frames.isFrameEmpty(self.id):
                    continue

            else:
                time.sleep(1/self.fps)
            self.frame_no += 1
            
        # Indicates that stream is done running
        self.shared_frames.end_stream(self.id, self.fps)
        print(self.stream_file + " stream ended")
        capture.release()