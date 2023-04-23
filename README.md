# Visual Object Tracking

This project uses YOLOv8 and deepSORT algorithm to do visual object tracking.

## Installation

Python virtual environment is not required but highly recommended. First, make sure the Python version is at least 3.9.

```bash
$ python3 --version
Python 3.9.16
```

Then activate the virtual environment and install all the required packages.
```bash
$ pwd
/Users/username/src/VisualObjectTracking
$ python3 -m venv env
$ source env/bin/activate
$ pip install --upgrade pip setuptools wheel
$ pip install -r requirements.txt
```

## Usage

Put your interesting video in the folder and modify the following line in the ```object_tracker.py```

```python
video_path = "YOUR_VIDEO_PATH"
```
Then run this file.
```bash
$ python object_tracker.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)