EXPECTED_SIZE = 300
CONFIDENCE_THRESHOLD = 0.7
IN_SCALE_FACTOR = 0.007843
MEAN_VAL = 127.53
FOCAL_Y = 450
X_RES = 640
Y_RES = 360
FPS = 15
MAX_DISTANCE = 800  # cm

PROTOTXT_PATH = "model/MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "model/MobileNetSSD_deploy.caffemodel"
CLASS_NAMES = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor")
SEARCHED_CLASSES = ["person"]
