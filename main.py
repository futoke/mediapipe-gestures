# hypercorn main:app --bind 0.0.0.0:8005
import os
import sys
import json
import time
import base64
import asyncio
import logging
import websockets

import cv2
import requests

import numpy as np
import mediapipe as mp

from fastapi import FastAPI
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from logging.handlers import RotatingFileHandler


STREAM_URL = "ws://localhost:8002/websocket"
PATRIK_URL = "http://localhost:8000"
TRIGGERS_LIST_URL = f"{PATRIK_URL}/api/list_triggers_by_type/"
TRIGGER_FIRED_URL = f"{PATRIK_URL}/api/trigger_fired"

TRIGGER_GESTURE = [2, ]
RECOONECT_PERIOD = 10
GESTURE_DELAY = 2

MODEL = "gesture_recognizer.task"
NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

if not os.path.exists("logs"):
    os.makedirs("logs")

log_file = RotatingFileHandler(
    'logs/gestures.log',
    maxBytes=32768, 
    backupCount=16
)
log_console = logging.StreamHandler(sys.setLogRecordFactory)

logging.basicConfig(
    handlers=(log_file, log_console), 
    format='[%(asctime)s | %(levelname)s]: %(message)s', 
    datefmt='%m.%d.%Y %H:%M:%S',
    level=logging.INFO
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

app = FastAPI()
app.fifo_queue = asyncio.Queue(1)

recognition_result_list = []


def compare_triggers(gesture):
    response = requests.post(
        TRIGGERS_LIST_URL,
        json={"trigger_type": TRIGGER_GESTURE},
        timeout=1.5
    )

    triggers = []
    # while True:
    if response.status_code == 200:
        triggers = response.json()
            # break
    
    for trigger in triggers:
        trigger_id = trigger['id']
        # Не знаю, что писать сюда, у меня есть лэндмарки и названия жестов,
        # пишу как будто сравниваю полученное название с названием из триггера.
        known_gesture = trigger.get('gesture_landmarks')
        
        if gesture == known_gesture:
            logging.info(f'Trigger {trigger_id} fired.')
            requests.get(f"{TRIGGER_FIRED_URL}/{trigger_id}/")


def configure_recognizer():
    def save_result(
        result: vision.GestureRecognizerResult,
        unused_output_image: mp.Image,
        timestamp_ms: int,
    ):
        recognition_result_list.append(result)

    # Initialize the gesture recognizer model.
    base_options = python.BaseOptions(model_asset_path=MODEL)

    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        result_callback=save_result,
    )
    return vision.GestureRecognizer.create_from_options(options)


async def bg_worker():
    last_recognized = {"image": None, "gesture": None, "landmarks": None}
    recognizer = configure_recognizer()

    ws = None

    while True:
        try:
            ws = await websockets.connect(STREAM_URL, ping_interval=None)
            break
        except ConnectionRefusedError:
            logging.error(
                f"No connection to stream server ({STREAM_URL}). "
                f"Trying to reconnect every {RECOONECT_PERIOD} sec..."
            )
            await asyncio.sleep(RECOONECT_PERIOD)

    while True:
        try:
            msg = await ws.recv() 
            msg = json.loads(msg)
            
            img_b64 = msg["frame"]
            decoded_data = base64.b64decode(img_b64)
            np_data = np.frombuffer(decoded_data, np.uint8)
            
            image = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
            image = cv2.flip(image, 1)

            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Run gesture recognizer using the model.
            recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

            if recognition_result_list:
                # Get gesture classification results.
                if recognition_result_list[0].gestures:
                    # Always one hand, otherwise change last index (0).
                    gesture = recognition_result_list[0].gestures[0]
                    landmarks = recognition_result_list[0].hand_landmarks
                    category_name = gesture[0].category_name

                    # if category_name not in ("None", last_recognized["gesture"]):
                    if category_name != "None":    
                        try:
                            compare_triggers(category_name)
                        except (
                            ConnectionError,
                            ConnectionRefusedError,
                            requests.exceptions.ConnectionError
                        ):
                            logging.error(
                                f"No connection to Patrik server ({PATRIK_URL}). "
                                f"Trying to reconnect every {RECOONECT_PERIOD} sec..."
                            )
                            await asyncio.sleep(RECOONECT_PERIOD)
                            continue

                        last_recognized["image"] = img_b64
                        last_recognized["gesture"] = category_name
                        last_recognized["landmarks"] = landmarks

                        if app.fifo_queue.full():
                            await app.fifo_queue.get()
                        if app.fifo_queue.empty():
                            await app.fifo_queue.put(last_recognized)
                            logging.info(f"Recognize {last_recognized.get('gesture')}")
                        
                        # Небольшая задержка, чтобы убрать руку из кадра
                        await asyncio.sleep(GESTURE_DELAY)
                    
                recognition_result_list.clear()          
        except (
            websockets.exceptions.ConnectionClosedError,
            ConnectionRefusedError
        ):
            logging.error(
                f"Stream server connection lost ({STREAM_URL}). "
                f"Trying to reconnect every {RECOONECT_PERIOD} sec..."
            )
            await asyncio.sleep(RECOONECT_PERIOD)
            try:
                ws = await websockets.connect(STREAM_URL, ping_interval=None)
            except:
                continue

@app.on_event("startup")
async def start_db():
    asyncio.create_task(bg_worker())


@app.get("/gesture_info")
async def gesture_info():
    data = await app.fifo_queue.get()
    return {
        "gesture": data["gesture"],
        "landmarks": data["landmarks"],
        "image": data["image"]
    }