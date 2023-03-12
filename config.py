import pickle

REAL_TIME = False
HAS_DISPLAY = False
RES_DIR = "res"
MEDIA_DIR = "media"
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
    pickle.dump(model, open(REGRESSION_MODEL_PATH, "wb"))

def load_regression_model(model):
    return pickle.load(open(REGRESSION_MODEL_PATH, "rb"))
    