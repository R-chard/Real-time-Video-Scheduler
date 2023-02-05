from scheduler import Scheduler
import cv2

scheduler = Scheduler("media")

while True:
    frame = scheduler.get_frame()
    if frame:
        cv2.imshow("Frame", frame)
