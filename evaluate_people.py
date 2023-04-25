import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import csv
import motmetrics as mm
from ultralytics import YOLO
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet


def evaluate(truth, hypo, acc):
    distance = mm.distances.iou_matrix(truth, hypo, max_iou=0.7)
    print(distance)
    t = [i+1 for i in range(truth.shape[0])]
    h = [i+1 for i in range(hypo.shape[0])]
    acc.update(
        t,
        h,
        distance
    )


def main():
    max_cosine_distance = 0.7
    nn_budget = None
    
    model = YOLO("yolov8s.pt")
    encoder = gdet.create_box_encoder('model_data/mars-small128.pb', batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, n_init=1)
    acc = mm.MOTAccumulator(auto_id=True)
    
    Track_only = ['person']
    eval_boxes = {}
    with open('eval_boxes_people.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            temp = row[0].split(',')
            if temp[0] != 'frames':
                box = np.array(temp[1:]).astype(float)
                if temp[0] not in eval_boxes.keys():
                    eval_boxes[temp[0]] = []
                eval_boxes[temp[0]].append(box)
    
    path = ('eval_images_people')
    dir_list = os.listdir(path)
    dir_list = sorted(dir_list)
    
    count = 0
    for file in dir_list[:100]:
        count += 1
        print(count)
        if file[-4:] == '.jpg':
            img = cv2.imread(path + '/' + file)
            try:
                original_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            except:
                break
            
            height, width, _ = img.shape
            results = model(img)
            CLASS_DICT = results[0].names
            key_list = list(CLASS_DICT.keys()) 
            val_list = list(CLASS_DICT.values())

            # extract bboxes to boxes (x, y, width, height), scores and names
            boxes, scores, names = [], [], []
            for bbox in results[0].boxes.data:
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
            
            truth = np.array([])
            file_num = str(int(file[:-4]))
            if file_num in eval_boxes.keys():
                truth = np.array(eval_boxes[file_num])
            hypo = np.array(tracked_bboxes)
            if hypo.shape != (0,):
                hypo = np.delete(hypo, np.s_[4:], 1)
                hypo[:, 2] -= hypo[:, 0]
                hypo[:, 3] -= hypo[:, 1]
            '''
            if truth.shape != (0,):
                truth[:, 2] -= truth[:, 0]
                truth[:, 3] -= truth[:, 1]
            '''
            #hypo = boxes
            print(hypo)
            print(truth)
            evaluate(truth, hypo, acc)
            print('--------')
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


if __name__ == "__main__":
    main()
