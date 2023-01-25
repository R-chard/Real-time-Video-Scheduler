import os

class Scheduler:
    def __init__(self, stream_dir):
        streams = [None for _ in range(len(stream_dir))]

        captures = os.listdir(stream_dir)
        for i in range(len(streams)):
            streams[i] = (captures[i],0) # capture, priority

