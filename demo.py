from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8m.pt")
    # results = model.predict("2.png", save=True, conf=0.4)
    # cv2.imwrite("2.png", results)
    # print(results[0])
    # Load a model
    model = YOLO('yolov8m.pt')  # load an official detection model
    model.train(data='mot16.yaml', epochs=1, imgsz=640)
    # model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
    # model = YOLO('path/to/best.pt')  # load a custom model

    # Track with the model "./IMAGES/test.mp4"
    # results = model.track(source="./IMAGES/test.mp4", show=True, conf=0.4) 
    #results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True, tracker="bytetrack.yaml") 

if __name__ == "__main__":
    main()