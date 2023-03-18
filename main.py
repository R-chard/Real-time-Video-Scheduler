import sys
sys.path.insert(1,"./shared/details")
import config
import os
import time
from object_tracker import ObjectTracker

# Empties out result files for a new batch
def clearResFiles(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir,f))

# Returns a list of (video files,objects of interest) to test
def getVideoDescrs(desc_file):
    videoDesc = []
    with open(os.path.join(config.SHARED_DIR, config.DETAILS_DIR, desc_file)) as f:
        while f:
            line = f.readline().strip(",\n\r ")
            if not line:
                break
            if line[0] == "#":
                continue
            objs_interest_line = f.readline().strip(",\n\r ")
            if not objs_interest_line:
                break
            objs_interest = set(map(lambda x: int(x), objs_interest_line.split(",")))
            videoDesc.append((line,objs_interest))

    return videoDesc

def record_result(values, file_name, isActual):
    if not isActual:
        filepath = os.path.join(config.SHARED_DIR, config.RESULTS_DIR, "predicted_" + file_name)
    else:
        filepath = os.path.join(config.SHARED_DIR, config.RESULTS_DIR, file_name)
    
    content_file, change_file, disruption_file,total_value_file = filepath + "_content_value.csv", filepath+"_change_value.csv", filepath+"_disruption_value.csv", filepath+"total_value.csv"
    #value = util.calcValue(content_value, disrupted_value)
    files = [content_file, change_file, disruption_file, total_value_file]
    content_values, change_values, disruption_values,total_values = zip(*values)
    values = [content_values, change_values, disruption_values,total_values]

    for i in range(len(files)):
        with open(files[i], "w") as f:
            for j in range(len(values[0])):
                # if j < 5:
                #     value = sum(values[i][0:j+1])/(j+1)
                # else:
                #     value = sum(values[i][j-5+1:j+1])/5
                value = values[i][j]
                #f.write(str(j) + ", " + str(values[j][i]) + "\n")
                f.write(str(value) + "\n")
if __name__ == "__main__":
    videoDescriptions = getVideoDescrs("description")
    #clearResFiles(config.RES_DIR)

    # collect data if not real time
    if not config.REAL_TIME:
        # Collect data one video at a time
        for i in range(len(videoDescriptions)):
            objectTracker = ObjectTracker([videoDescriptions[i]])
            hasFrames = True
            while hasFrames:
                hasFrames = objectTracker.run()
            record_result(objectTracker.values, videoDescriptions[i][0], True)
            record_result(objectTracker.scheduler.predicted_values, videoDescriptions[i][0], False)
    else:
        objectTracker = ObjectTracker(videoDescriptions)
        hasFrames = True
        while hasFrames:
            hasFrames = objectTracker.run()
        objectTracker.record_result()

    print("All streams terminated")
    #except Exception as e:
    #    print(str(e))
    #finally:
    objectTracker.cleanUp()