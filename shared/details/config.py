import pickle
import os

# File contains configuration for the simulation

REAL_TIME = False       # which mode the scheduler should run in
HAS_DISPLAY = False     # displays a window if True
FV_OPTIMISATION = False # whether we are using the optimisation version
RUN_ALL_MODES = True
SCHEDULER_MODE = "fv"   # can be fv, rr or edf
SHARED_DIR = "shared"
DETAILS_DIR = "details"
MEDIA_DIR = "media"
RESULTS_DIR = "output"
GRAPHS_DIR = "graph"
FPS_SCALE = 1 / 10
SCREEN_MAX_WIDTH = 1280
SCREEN_MAX_HEIGHT = 720
REGRESSION_MODEL_PATH = "multiple_regression.sav"

def get_resized_dim(frame):
    width = int(frame.shape[1])
    height = int(frame.shape[0])

    if width> SCREEN_MAX_WIDTH:
        height = round(height* SCREEN_MAX_WIDTH/width)
        width = SCREEN_MAX_WIDTH

    if height> SCREEN_MAX_HEIGHT:
        width = round(width* SCREEN_MAX_HEIGHT/height)
        height = SCREEN_MAX_HEIGHT
    
    return (width,height)

def save_regression_model(model):
    file_path = os.path.join(SHARED_DIR, DETAILS_DIR, REGRESSION_MODEL_PATH)
    pickle.dump(model, open(file_path, "wb"))

def load_regression_model():
    file_path = os.path.join(SHARED_DIR, DETAILS_DIR, REGRESSION_MODEL_PATH)
    return pickle.load(open(file_path, "rb"))

def get_csv_paths(vid_file, is_prediction):
    files = [vid_file[:-4] + "_content.csv",
        vid_file[:-4] + "_dynamism.csv",
        vid_file[:-4] + "_impact.csv",
        vid_file[:-4] + "_total.csv"]
    if is_prediction:
        files = ["predicted_" + _file for _file in files]
    files = [os.path.join(SHARED_DIR, RESULTS_DIR, _file) for _file in files]
    
    return files

def get_plot_path(path):
    return os.path.join(SHARED_DIR, GRAPHS_DIR,path)

    