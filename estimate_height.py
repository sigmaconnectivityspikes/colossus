from collections import deque
import cv2
import numpy as np
import pyrealsense2 as rs
import settings


class CircularBuffer(deque):
    def __init__(self, size=0):
             super(CircularBuffer, self).__init__(maxlen=size)
    @property
    def average(self):
        return sum(self) / len(self)

    @property
    def median(self):
        return np.median(self)

    @property
    def data(self):
        return list(self)


def configure_stream(x_res, y_res, fps):
    """
    Configure depth and color streams.
    """
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, x_res, y_res, rs.format.z16, fps)
    rs_config.enable_stream(rs.stream.color, x_res, y_res, rs.format.bgr8, fps)
    return pipeline, rs_config


def run_stream(pipeline, config):
    profile = pipeline.start(config)
    return profile


def add_bounding_box(image, x_min, y_min, x_max, y_max, color):
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)


def add_text(image, x_min, y_min, color, label):
    """
    Add text to image.
    """
    y = y_min - 15 if y_min - 15 > 15 else y_min + 15
    cv2.putText(image, label, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)


def get_distance_to_object(depth_image, x_min, y_min, x_max, y_max, depth_scale):
    """
    Use more points to calculate reliable distance from person.
    """
    px_width = x_max - x_min
    px_height = y_max - y_min
    x_min = int(x_min + px_width / 4)
    x_max = int(x_max - px_width / 4)
    y_min = int(y_min + px_height / 3)
    y_max = int(y_max - px_height / 6)
    depth = depth_image[y_min:y_max, x_min:x_max].astype(float)
    depth = depth * depth_scale
    depth_slice = depth[(depth < np.quantile(depth, 0.7)) & (depth > np.quantile(depth, 0.3))]
    if depth_slice.size:
        distance = np.mean(depth_slice)
        distance *= 100  # from m to cm
        return distance
    return None


def preprocess_image(image, expected_size, in_scale_factor, mean_val):
    """
    Create blob from image as an input to neural network.
    """
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (expected_size, expected_size)),
                                 in_scale_factor, (expected_size, expected_size), mean_val)
    return blob


def calculate_height(distance, y_max, y_min, focal_y):
    """
    Calculate real person height in centimeters.
    """
    px_height = y_max - y_min
    person_height = distance * px_height / focal_y
    return person_height


def process_detections(detections, depth_image, height_buffer, depth_scale, confidence_threshold,
                       x_res, y_res, class_names, searched_classes, max_distance, focal_y):
    """
    Get information about all objects.
    :return:
    """
    detected_objects_parameters = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7]
            box[box < 0] = 0
            box = box * np.array([x_res, y_res, x_res, y_res])
            (x_min, y_min, x_max, y_max) = box.astype("int")
            label = class_names[idx]

            if label in searched_classes:
                distance = get_distance_to_object(depth_image, x_min, y_min, x_max, y_max,
                                                  depth_scale)
                if not distance or distance > max_distance:
                    continue
                person_height = calculate_height(distance, y_max, y_min, focal_y)
                height_buffer.append(person_height)
                person_averaged_height = height_buffer.median
                detected_objects_parameters.append([label, idx, confidence, x_min, x_max, y_min,
                                                    y_max, distance, person_averaged_height])
    return detected_objects_parameters


def main():
    # Get settings parameters
    expected_size = settings.EXPECTED_SIZE
    confidence_threshold = settings.CONFIDENCE_THRESHOLD
    in_scale_factor = settings.IN_SCALE_FACTOR
    mean_val = settings.MEAN_VAL
    focal_y = settings.FOCAL_Y
    x_res = settings.X_RES
    y_res = settings.Y_RES
    fps = settings.FPS
    max_distance = settings.MAX_DISTANCE
    prototxt_path = settings.PROTOTXT_PATH
    model_path = settings.MODEL_PATH
    class_names = settings.CLASS_NAMES
    searched_classes = settings.SEARCHED_CLASSES

    # Read neural network model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))
    height_buffer = CircularBuffer(size=15)
    pipeline, rs_config = configure_stream(x_res, y_res, fps)
    '''
    # Get OEM device information, not used because of dynamic calibration
    cfg = pipeline.start(rs_config)
    intr = profile.as_video_stream_profile().get_intrinsics()
    focal_x = intr.fx
    '''
    # Run stream from realsense camera
    profile = run_stream(pipeline, rs_config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            # Align RGB and depth frames to match corresponding points
            aligned_frames = align.process(frames)
            # Get RGB frame
            color_frame = aligned_frames.first(rs.stream.color)
            # Get depth frame
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            blob = preprocess_image(color_image, expected_size, in_scale_factor, mean_val)
            net.setInput(blob, "data")
            # Detect objects on RGB image
            detections = net.forward("detection_out")
            # Get label, position, distance, height and other objects information
            objects_parameters =  process_detections(detections, depth_image, height_buffer,
                                                     depth_scale, confidence_threshold, x_res,
                                                     y_res, class_names, searched_classes,
                                                     max_distance, focal_y)
            # Apply colormap to depth image
            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                            cv2.COLORMAP_JET)

            # Put information on image
            for (label, idx, confidence, x_min, x_max, y_min, y_max, distance,
                 height) in objects_parameters:
                label_text = "{} Height: {:.2f} Distance: {:.1f}".format(label, height, distance)
                color = colors[idx]
                add_bounding_box(color_image, x_min, y_min, x_max, y_max, color)
                add_text(color_image, x_min, y_min, color, label_text)
                add_bounding_box(depth_image, x_min, y_min, x_max, y_max, color)
                add_text(depth_image, x_min, y_min, color, label_text)

            # Display RGB and depth frames
            cv2.imshow("Color", color_image)
            cv2.imshow("Depth", depth_image)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
