import cv2
import sys
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from object_detection import ObjectDetection
from multiprocessing.managers import BaseManager
from scheduler import Scheduler
from shared_frames import SharedFrames

if __name__ == '__main__':
    # Initialize Object Detection
    od = ObjectDetection()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    multiTracker = cv2.legacy.MultiTracker_create()

    # Filter out invalid files
    stream_files = []
    stream_dir = "MEDIA"
    for f in os.listdir(stream_dir):
        vid_path = os.path.join(stream_dir,f)
        if not os.path.isfile(vid_path) or not os.path.splitext(vid_path)[1] == ".mp4":
            continue
        stream_files.append(vid_path)

    # Init shared frames
    BaseManager.register('SharedFrames', SharedFrames)
    manager = BaseManager()
    manager.start()
    frames = manager.SharedFrames(len(stream_files))

    # Init scheduler
    scheduler = Scheduler(stream_files,frames)

    frameID = 0
    object_id = 0

    while True:
        stream_id, frame = scheduler.get_frame()
        if frame.size == 0:
            continue
        #fg_mask = bg_subtractor.apply(frame)
        #frame = cv2.merge([fg_mask,fg_mask,fg_mask])
        success,tracked_bboxes = multiTracker.update(frame)
        frameID += 1
        #if frameID > 3 and frameID%10 != 0:
        #    continue
        
        tracked = {}

        # Print all previously tracked objects
        for bbox in tracked_bboxes:
            x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0), 2, 1)
            object_id += 1

        # Object detection
        (class_ids, scores, bboxes) = od.detect(frame)
        for bbox in bboxes:
            x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            
            centre_x = (x+x+w)/2
            centre_y = (y+y+h)/2

            found = False
            for t_bbox in tracked_bboxes:
                tx,ty,tw,th = int(t_bbox[0]),int(t_bbox[1]),int(t_bbox[2]),int(t_bbox[3])

                centre_tx = (tx+tx+w)/2
                centre_ty = (ty+ty+h)/2

                if tx < centre_x < tx+tw and ty < centre_y < ty+th or x < centre_tx < x+w and y < centre_ty < y+h:
                    found = True
                    break

            if not found:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255), 2, 1)
                tracker = cv2.legacy.TrackerKCF_create()
                multiTracker.add(tracker,frame,bbox)
                object_id += 1

        cv2.putText(frame, "Frame: " + str(frameID),(50,50),0,1,(0,0,255), 2)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
