import sys
import argparse
import pathlib
sys.path.insert(1, str(pathlib.Path.cwd().parents[0]) + "/common")

import cv2
import utils as util
import time
import numpy as np

def main(sources):
    # Read image from the video source
    vs = cv2.VideoCapture("/home/snucse/AI-based-Traffic-Control-System--/datas/video1.mp4")

    # Check if the video was opened successfully
    if not vs.isOpened():
        print(f"Error: Unable to open video source {sources[0]}")
        return

    # Create a network given yolov5s model
    net = cv2.dnn.readNet("/home/snucse/AI-based-Traffic-Control-System--/models/yolov5s.onnx")

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("2")
    ln = net.getUnconnectedOutLayersNames()  # returns the name of output layer

    # Initial configuration of each lane's order
    lanes = util.Lanes([util.Lane(0, None, 1), util.Lane(0, None, 3), util.Lane(0, None, 4), util.Lane(0, None, 2)])

    wait_time = 0

    while True:
        # Read the next frame from the video source
        success, frame = vs.read()

        # If the frame was not successfully captured, then we have reached the end of the stream or there is a disconnection
        if not success:
            print("Reached the end of the video or error in reading the frame.")
            break

        # Assign the frame only to lane 1 (since we only have one video feed)
        for lane in lanes.getLanes():
            if lane.lane_number == 1:
                lane.frame = frame  # Process lane 1
            else:
                lane.frame = None  # Set frame to None for other lanes to avoid errors

        # Process the lanes with available frames
        start = time.time()
        lanes = util.final_output(net, ln, lanes)  # returns lanes object with processed frame
        end = time.time()
        print("Total processing time: " + str(end - start))

        if wait_time <= 0:
            images_transition = util.display_result(wait_time, lanes)
            final_image = cv2.resize(images_transition, (1020, 720))
            cv2.imshow("f", final_image)
            cv2.waitKey(100)

            wait_time = util.schedule(lanes)  # returns waiting duration of each lane

        images_scheduled = util.display_result(wait_time, lanes)
        final_image = cv2.resize(images_scheduled, (1020, 720))
        cv2.imshow("f", final_image)
        cv2.waitKey(1)
        wait_time = wait_time - 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determines duration based on car count on images")
    parser.add_argument("--s", help="video feeds to be inferred on, the videos must reside in the datas folder", type=str, default=r"video1.mp4")

    args = parser.parse_args()
    sources = args.s.split(",")
    main(sources)

