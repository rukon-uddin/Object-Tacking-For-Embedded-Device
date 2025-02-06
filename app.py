from flask import Flask, render_template, Response, redirect, url_for
import cv2
import time
import numpy as np
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow as tf
from core.utils import format_boxes, read_class_names
from core.yolov4 import filter_boxes
from tools import generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import preprocessing, nn_matching

# Initialize Flask app
app = Flask(__name__)

# Global variables to manage video feed and model
camera = cv2.VideoCapture(0)
output_frame = None

# Model Initialization
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

interpreter = tf.lite.Interpreter(model_path='./checkpoints/yolov4-tiny-416.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = 416

# Initialize DeepSORT tracking
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

class_names = read_class_names('./data/classes/coco.names')

@app.route('/')
def index():
    return render_template('index.html')


def model_inference(image_input):
    interpreter.set_tensor(input_details[0]['index'], image_input)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.45, input_shape=tf.constant([input_size, input_size]))
    return boxes, pred_conf


def process_frame():
    global output_frame, camera
    success, frame = camera.read()
    if not success:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (input_size, input_size))
    image_data = frame_resized / 255.0
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    boxes, pred_conf = model_inference(image_data)
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.45
    )

    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0][:num_objects]
    scores_npy = scores.numpy()[0][:num_objects]
    classes_npy = classes.numpy()[0][:num_objects]

    original_h, original_w, _ = frame.shape
    bboxes = format_boxes(bboxes, original_h, original_w)

    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores_npy, [class_names[int(c)] for c in classes_npy], features)]
    indices = preprocessing.non_max_suppression(np.array([d.tlwh for d in detections]), np.array([d.class_name for d in detections]), nms_max_overlap, np.array([d.confidence for d in detections]))
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)

    return frame


def generate_frames():
    while True:
        frame = process_frame()
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_feed')
def start_feed():
    return redirect(url_for('index'))


@app.route('/stop_feed')
def stop_feed():
    global camera
    camera.release()
    camera = cv2.VideoCapture(0)
    return redirect(url_for('index'))


if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        camera.release()
        session.close()
