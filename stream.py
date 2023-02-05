from multiprocessing import Process
import cv2
import time

class Stream(Process):
    def __init__(self,id,stream_file, shared_frames):
        super(Stream,self).__init__()
        self.id = id
        self.stream_file = stream_file
        self.shared_frames = shared_frames

    def run(self):
        capture = cv2.VideoCapture(self.stream_file)
        fps = capture.get(cv2.CAP_PROP_FPS)

        while capture.isOpened():
            success,frame = capture.read()
            if not success:
                break
            self.shared_frames.set_frame(self.id, frame)
            time.sleep(1/fps)
        capture.release()