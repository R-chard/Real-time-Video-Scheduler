# import cv2
# from object_detection import ObjectDetection
# import config

# capture = cv2.VideoCapture("media/full_traffic.mp4")
# od = ObjectDetection()

# while capture.isOpened():
#     success,frame = capture.read()
#     if not success:
#         break
#     dim = config.get_resized_dim(frame)
#     frame = cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)
    
#     #obj_ids, confidences, bboxes = od.detect(frame)
#     #detected = []
#     obj_ids = 
#     detected = od.detect(frame)
#         for i in range(len(detected[0])):
#             if obj_ids[i] == 2:
#                 detected.append((obj[0][i],obj[1][i],obj[2][i]))

#     print(detected)
