# Run individually to extract frame given frame number
import cv2
import sys
import os
import config

# To use: python3 frame_extractor.py <stream name> <frame_number>
if __name__ == "__main__":

    stream, frame_no = sys.argv[1], int(sys.argv[2])
    stream_path = os.path.join("./media",stream)

    cap = cv2.VideoCapture(stream_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Resize frame
        dim = config.get_resized_dim(frame)
        frame = cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)
        cv2.putText(frame, "Frame: " + str(frame_no),(50,50),0,1,(0,0,255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0)
        if key == 27:
            break
        frame_no += 1
    cap.release()