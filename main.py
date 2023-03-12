import sys
import os
import config
from object_tracking import ObjectTracker

# Empties out result files for a new batch
def clearResFiles(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir,f))

# Returns a list of (video files,objects of interest) to test
def getVideoDescrs(dir):
    videoDesc = []
    with open(os.path.join(dir,"description")) as f:
        while f:
            line = f.readline().strip(",\n\r ")
            if not line:
                break
            if line[0] == "#":
                continue
            vid = os.path.join(dir,line)
            objs_interest_line = f.readline().strip(",\n\r ")
            if not objs_interest_line:
                break
            objs_interest = set(map(lambda x: int(x), objs_interest_line.split(",")))
            videoDesc.append((vid,objs_interest))

    return videoDesc

# Check that user input is valid
def processUserInput():
    if(len(sys.argv)!= 3 or sys.argv[1] not in ["gpu", "desktop"] or sys.arv[2] not in ["rt", "offline"]):
        print("Usage: python object_tracking.py <gpu or desktop> <rt or offline>")
        exit(1)
    return (sys.argv[1] != "gpu", sys.arv[2] == rt)

if __name__ == "__main__":
    config.HAS_DISPLAY, config.REAL_TIME = processUserInput()
    videoDescriptions = getVideoDescrs(config.MEDIA_DIR)
    clearResFiles(config.RES_DIR)

    #try:
    # collect data if not real time
    if not config.REAL_TIME:
        # Collect data one video at a time
        for videoDescription in videoDescriptions:
            objectTracker = ObjectTracker([videoDescription])
            hasFrames = True
            while hasFrames:
                hasFrames = objectTracker.run()
            objectTracker.record_result()
    else:
        objectTracker = ObjectTracker(videoDescriptions)
        hasFrames = True
        while hasFrames:
            hasFrames = objectTracker.run()

    print("All streams terminated")
    #except Exception as e:
    #    print(str(e))
    #finally:
    objectTracker.cleanUp()