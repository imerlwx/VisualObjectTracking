import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
import time

from utils import draw_bbox
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from ultralytics import YOLO


def Object_tracking(model, video_path, output_path, show=False, Track_only = []):
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    model : yolo model
        YOlOv8 model in this task
    video_path, output_path : string 
        Video path to the tracking task.
    show : bool, optional
        Whether to show the video after tracking.
    Track_only : Optional[ndarray]
        A list of objects to track.

    """
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None

    # initialize deep sort object
    # mars-small128.pb is a CNN model that without the last fc layer
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break

        t1 = time.time()
        results = model(frame)
        CLASS_DICT = results[0].names
        key_list = list(CLASS_DICT.keys()) 
        val_list = list(CLASS_DICT.values())
        t2 = time.time()

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in results[0].boxes.data:
            # bbox = result.boxes.data
            # print(bbox)
            if len(Track_only) !=0 and CLASS_DICT[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1])])
                scores.append(bbox[4])
                names.append(CLASS_DICT[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, CLASS_DICT, tracking=True)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()


video_path   = "cardemo2.mp4"
model = YOLO("yolov8s.pt")
Object_tracking(model, video_path, "detection.mp4", show=True, Track_only = ["car"])
