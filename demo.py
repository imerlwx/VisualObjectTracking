from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    results = model("IMAGES/bus.jpg")
    print(results[1].boxes.data)

if __name__ == "__main__":
    main()