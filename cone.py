import time
import cv2
import numpy as np
import pyrealsense2 as rs
from utils import cv_utils
from utils import operations as ops
import serial
import tensorflow as tf

FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'
TFLITE_MODEL_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.tflite'
OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5

# Modes
MODE_MANUAL = 0
MODE_AUTOMATIC = 1

ser = serial.Serial('COM6', 9600, timeout=1)
ser.flush()

SPEED = 0
DIRECTION = 30

def writeArduino(d, s):
    try:
        ACTION = (str(d) + "#" + str(s) + "\n").encode('utf-8')
        ser.write(ACTION)
        line = ser.readline().decode('utf-8').rstrip()
    except Exception as e:
        print(e)

def convert_to_tflite(model_path, tflite_model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

def draw_circle(image, center, radius, color, thickness=1):
    cv2.circle(image, center, radius, color, thickness)

def draw_vertical_line(image, start_point, end_point, color, thickness=10):
    cv2.line(image, start_point, end_point, color, thickness)

def main():
    # Convert TensorFlow model to TensorFlow Lite
    convert_to_tflite(FROZEN_GRAPH_PATH, TFLITE_MODEL_PATH)

    # Read video from disk and count frames
    cap = cv2.VideoCapture(2)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # CROP_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # CROP_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH) as interpreter:
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        processed_images = 0
        prev_orange_cone = None  # Store previous frame's largest orange cone
        prev_green_cone = None   # Store previous frame's largest green cone

        mode = MODE_MANUAL  # Initial mode is manual
        mode_display_text = "Mode: MANUAL"

        while cap.isOpened():

            if DETECT_EVERY_N_SECONDS:
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        processed_images * fps * DETECT_EVERY_N_SECONDS)

            ret, frame = cap.read()
            if ret:
                tic = time.time()

                # crops are images as ndarrays of shape
                # (number_crops, CROP_HEIGHT, CROP_WIDTH, 3)
                # crop coordinates are the ymin, xmin, ymax, xmax coordinates in
                # the original image
                crops, crops_coordinates = ops.extract_crops(
                    frame, CROP_HEIGHT, CROP_WIDTH,
                    CROP_STEP_VERTICAL, CROP_STEP_VERTICAL)

                # Uncomment this if you also uncommented the two lines before
                # creating the TF session.
                # crops = np.array([crops[0]])
                # crops_coordinates = [crops_coordinates[0]]

                interpreter.set_tensor(input_details[0]['index'], crops)
                interpreter.invoke()

                detection_dict = {}
                for output_detail in output_details:
                    tensor_data = interpreter.get_tensor(output_detail['index'])
                    detection_dict[output_detail['name']] = tensor_data

                # The detection boxes obtained are relative to each crop. Get
                # boxes relative to the original image
                # IMPORTANT! The boxes coordinates are in the following order:
                # (ymin, xmin, ymax, xmax)
                boxes = []
                for box_absolute, boxes_relative in zip(
                        crops_coordinates, detection_dict['detection_boxes']):
                    boxes.extend(ops.get_absolute_boxes(
                        box_absolute,
                        boxes_relative[np.any(boxes_relative, axis=1)]))
                if boxes:
                    boxes = np.vstack(boxes)

                # Remove overlapping boxes
                boxes = ops.non_max_suppression_fast(
                    boxes, NON_MAX_SUPPRESSION_THRESHOLD)

                # Get scores to display them on top of each detection
                boxes_scores = detection_dict['detection_scores']
                boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

                orange_cones = []
                green_cones = []

                for box, score in zip(boxes, boxes_scores):
                    if score > SCORE_THRESHOLD:
                        ymin, xmin, ymax, xmax = box
                        color_detected_rgb = cv_utils.predominant_rgb_color(
                            frame, int(ymin), int(xmin), int(ymax), int(xmax))
                        text = '{:.2f}'.format(score)
                        if cv_utils.get_color_name(color_detected_rgb) == "orange":
                            orange_cones.append((int(ymin), int(xmin), int(ymax), int(xmax), color_detected_rgb, text))
                        elif cv_utils.get_color_name(color_detected_rgb) == "green":
                            green_cones.append((int(ymin), int(xmin), int(ymax), int(xmax), color_detected_rgb, text))

                # Show the largest orange cone
                if len(orange_cones) > 0:
                    largest_orange_cone = max(orange_cones, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]))
                    ymin, xmin, ymax, xmax, color_detected_rgb, text = largest_orange_cone
                    cv_utils.add_rectangle_with_text(
                        frame, ymin, xmin, ymax, xmax,
                        color_detected_rgb, text)

                # Show the largest green cone
                if len(green_cones) > 0:
                    largest_green_cone = max(green_cones, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]))
                    ymin, xmin, ymax, xmax, color_detected_rgb, text = largest_green_cone
                    cv_utils.add_rectangle_with_text(
                        frame, ymin, xmin, ymax, xmax,
                        color_detected_rgb, text)

                if OUTPUT_WINDOW_WIDTH:
                    frame = cv_utils.resize_width_keeping_aspect_ratio(
                        frame, OUTPUT_WINDOW_WIDTH)

                # Get frame dimensions
                frame_height, frame_width, _ = frame.shape

                # Draw a circle in the center of the frame
                center = (int(frame_width / 2), int(frame_height / 2))
                radius = 30
                circle_color = (0, 255, 0)  # Green color
                circle_thickness = 20
                draw_circle(frame, center, radius, circle_color, circle_thickness)

                # Draw vertical line if both orange and green cones are detected
                if len(orange_cones) > 0 and len(green_cones) > 0:
                    largest_orange_cone = max(orange_cones, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]))
                    largest_green_cone = max(green_cones, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]))
                    center_x_original = (largest_orange_cone[1] + largest_orange_cone[3] + largest_green_cone[1] + largest_green_cone[3]) / 4
                    center_x = int(center_x_original * OUTPUT_WINDOW_WIDTH / frame_width)
                    start_point = (center_x, 0)  # Start from the top of the image
                    end_point = (center_x, frame_height)  # End at the bottom of the image

                    if mode == MODE_AUTOMATIC:  # Only in automatic mode
                        if center_x < center[0] - radius:
                            print("Go right")
                            writeArduino(0, 1)
                        elif center_x > center[0] + radius:
                            print("Go left")
                            writeArduino(45, 1)
                        else:
                            print("Go straight")
                            writeArduino(30, 1)

                    draw_vertical_line(frame, start_point, end_point, (0, 0, 255))  # Red color

                # Display the current mode on the screen
                cv2.putText(frame, mode_display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow('Detection result', frame)
                cv2.waitKey(1)
                processed_images += 1

                toc = time.time()
                processing_time_ms = (toc - tic) * 100
                # logging.debug(
                #     'Detected {} objects in {} images in {:.2f} ms'.format(
                #         len(boxes), len(crops), processing_time_ms))

                # Check for mode switch (M) and toggle between manual and automatic modes
                key = cv2.waitKey(1)
                if key & 0xFF == ord('m') or key & 0xFF == ord('M'):
                    mode = MODE_AUTOMATIC if mode == MODE_MANUAL else MODE_MANUAL
                    mode_display_text = "Mode: AUTOMATIC" if mode == MODE_AUTOMATIC else "Mode: MANUAL"

                # Manual control keys
                if mode == MODE_MANUAL:
                    if key & 0xFF == ord('w') or key & 0xFF == ord('W'):
                        print("Go straight")
                    elif key & 0xFF == ord('s') or key & 0xFF == ord('S'):
                        print("Stop")
                    elif key & 0xFF == ord('a') or key & 0xFF == ord('A'):
                        print("Go left")
                    elif key & 0xFF == ord('d') or key & 0xFF == ord('D'):
                        print("Go right")

                if key & 0xFF == ord('q'):
                    break

            else:
                # No more frames. Break the loop
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
