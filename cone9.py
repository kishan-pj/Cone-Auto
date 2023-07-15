from __future__ import division
import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils

FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'
OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


def draw_circle(image, center, radius, color, thickness=1):
    cv2.circle(image, center, radius, color, thickness)


def draw_vertical_line(image, start_point, end_point, color, thickness=10):
    cv2.line(image, start_point, end_point, color, thickness)


def main():
    # Read TensorFlow graph
    detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)

    # Read video from disk and count frames
    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # CROP_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # CROP_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tf.Session(graph=detection_graph) as sess:

        processed_images = 0
        prev_orange_cone = None  # Store previous frame's largest orange cone
        prev_green_cone = None   # Store previous frame's largest green cone

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

                detection_dict = tf_utils.run_inference_for_batch(crops, sess)

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
                radius = 40
                circle_color = (0, 255, 0)  # Green color
                circle_thickness = 20
                draw_circle(frame, center, radius, circle_color, circle_thickness)

                # Draw vertical line if both orange and green cones are detected
                if len(orange_cones) > 0 and len(green_cones) > 0:
                    largest_orange_cone = max(orange_cones, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]))
                    largest_green_cone = max(green_cones, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]))
                    center_x = int((largest_orange_cone[1] + largest_orange_cone[3] + largest_green_cone[1] + largest_green_cone[3]) / 4)
                    start_point = (center_x, 0)  # Start from the top of the image
                    end_point = (center_x, frame_height)  # End at the bottom of the image
                    draw_vertical_line(frame, start_point, end_point, (0, 0, 255))  # Red color

                    # Determine the position of the line relative to the circle
                    if center_x < center[0] - radius:
                        print("Go right")
                    elif center_x > center[0] + radius:
                        print("Go left")
                    else:
                        print("Go straight")

                cv2.imshow('Detection result', frame)
                cv2.waitKey(1)
                processed_images += 1

                toc = time.time()
                processing_time_ms = (toc - tic) * 100
                # logging.debug(
                #     'Detected {} objects in {} images in {:.2f} ms'.format(
                #         len(boxes), len(crops), processing_time_ms))

                # Check for quit key (q) and exit the loop
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            else:
                # No more frames. Break the loop
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
