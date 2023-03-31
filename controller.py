import sys
sys.path.insert(1,"./shared/details")
import config
import os
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
from client import Client
from matplotlib.ticker import MaxNLocator

# Entry point of the program. Initiates the simulation of the real-time scheduler and runs it. 
# Handles majority of output to terminal and formatting the results into graph

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

# record calculated values to csv
def record_result_values(file_name, values, predicted_vals):
    file_sets = [config.get_csv_paths(file_name,False),config.get_csv_paths(file_name,True)]
    to_write = values
    for files in file_sets:
        for i in range(len(files)):
            with open(files[i], "w") as f:
                for j in range(1,len(to_write)):
                    # 0th frame ignored
                    f.write(str(to_write[j][i]) + "\n")
        to_write = predicted_vals

# Output results to terminal and graphs
def output_performance(client, total_time):
    s = client.scheduler
    print("\nResults for scheduler running in: " + config.SCHEDULER_MODE)
    print("Decision accuracy: " + str(s.get_decision_accuracy()))
    print("Total frames processed: " + str(s.total_frames_processed))
    frames_missed = s.get_total_frames() - s.total_frames_processed
    print("Total frames dropped: " + str(frames_missed))
    print("Frames dropped/ second: " + str(frames_missed/total_time))

    choice_times = np.array(client.choice_times)
    print("Decision time avg: " + str(np.average(choice_times)) + ", std deviation: " + str(np.std(choice_times)))
    application_times = np.array(client.application_times)
    decision_time_pct = np.divide(choice_times,application_times)*100
    print("Decision time percentage: " + str(np.average(decision_time_pct)) +", std deviation: " + str(np.std(decision_time_pct)))
    print("JFI: " + str(s.jfi))

    cpu_loads = np.array(client.cpu_loads)
    cpu_memory = np.array(client.cpu_memory)

    print("CPU usage: \n\tAverage: " + str(np.average(cpu_loads)) + "\n\tMax: " + str(cpu_loads.max()) + "\n\tMin: " + str(cpu_loads.min()))
    print("CPU memory used: \n\tAverage: " + str(np.average(cpu_memory)) + "\n\tMax: " + str(cpu_memory.max()) + "\n\tMin: " + str(cpu_memory.min()))

    # -- Avg decision time against total number of objs in stream --"
    obj_choice_time = {}
    for i in range(len(client.total_obj)):
        obj_count = client.total_obj[i]
        if not obj_count:
            continue
        if obj_count not in obj_choice_time:
            obj_choice_time[obj_count] = [0,0]
        obj_choice_time[obj_count][0] += client.choice_times[i]
        obj_choice_time[obj_count][1] += 1
    
    obj_vals = [(int(k),v[0]/v[1]) for k,v in obj_choice_time.items()]
    obj_vals.sort(key = lambda x:x[0])
    x_vals,y_vals = zip(*obj_vals)
    obj_choice_x = np.array(list(x_vals))
    obj_choice_y = np.array(list(y_vals))

    # -- Avg decision time against total number of streams --"
    stream_choice_time = {}
    
    for i in range(len(client.stream_count)):
        stream_count = client.stream_count[i]
        if stream_count not in stream_choice_time:
            stream_choice_time[stream_count] = [0,0]
        stream_choice_time[stream_count][0] += client.choice_times[i]
        stream_choice_time[stream_count][1] += 1

    stream_vals = [(int(k),v[0]/v[1]) for k,v in stream_choice_time.items()]
    stream_vals.sort(key = lambda x:x[0])
    x_vals,y_vals = zip(*stream_vals)
    stream_choice_x = np.array(list(x_vals))
    stream_choice_y = np.array(list(y_vals))

    return (choice_times, decision_time_pct, cpu_loads, cpu_memory,(obj_choice_x, obj_choice_y), (stream_choice_x, stream_choice_y))

# Compare the results obtained through all scheduling algorithms
def compare_all(*res):

    line_labels = ["rr", "edf", "fv", "fv-opt"]
    titles = ["choice.png", "choice_pct.png", "cpu_load.png", "cpu_memory.png", "time_per_obj.png", "time_per_stream.png"]
    y_labels = ["Decision time(s)", "Decision time as a % of processing time(%)","CPU Load(%)", "CPU Memory Usage(%)", 
    "Decision time(s)", "Decision time(s)"]
    x_labels = ["Total number of objects in frames", "Total number of active streams"]
    for j in range(len(res[0])):
        plt.ylabel(y_labels[j])

        for i in range(len(res)):
            if len(res[i][j]) != 2:
                plt.xlabel("Frames processed")
                x_points = np.arange(0,len(res[i][j]),dtype="int")
                y_points = res[i][j]
            else:
                plt.xlabel(x_labels[j-4])
                x_points, y_points = res[i][j]
            plt.plot(x_points, y_points,label=line_labels[i])
        plt.legend()
        plt.savefig(config.get_plot_path(titles[j]))
        plt.clf()

if __name__ == "__main__":
    def run_eval_rt(mode):
        config.SCHEDULER_MODE = mode
        full_mode = mode
        if mode == "fv":
            if config.FV_OPTIMISATION:
                full_mode += " with optimisation"
            else:
                full_mode += " without optimisation"
        print("\n----- Running scheduler under in: " + full_mode + " -----\n")
        client = Client(videoDescriptions)
        client.start_scheduler()
        hasFrames = True
        start = time.process_time()
        while hasFrames:
            hasFrames = client.run()
        end = time.process_time()
        client.cleanUp()
        return output_performance(client, end-start)

    videoDescriptions = getVideoDescrs("description")

    # collect data if not real time
    if not config.REAL_TIME:
        # Collect data one video at a time
        for file_name,objs_of_interest in videoDescriptions:
            client = Client([[file_name,objs_of_interest]])
            client.start_scheduler()
            hasFrames = True
            while hasFrames:
                hasFrames = client.run()

            evaluator_vals, predicted_vals = client.get_values()
            record_result_values(file_name, evaluator_vals, predicted_vals)
            client.cleanUp()
    else:
        if config.RUN_ALL_MODES:
            res_rr = run_eval_rt("rr")
            gc.collect()
            res_edf = run_eval_rt("edf")
            gc.collect()
            config.FV_OPTIMISATION = False
            res_fv_base = run_eval_rt("fv")
            gc.collect()
            config.FV_OPTIMISATION = True
            res_fv_better = run_eval_rt("fv")
            compare_all(res_rr, res_edf, res_fv_base, res_fv_better)
        else:
            run_eval_rt(config.SCHEDULER_MODE)
            
    print("All streams terminated")