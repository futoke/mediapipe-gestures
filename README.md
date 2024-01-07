Код взят отсюда:

```
https://github.com/googlesamples/mediapipe/tree/main/examples/gesture_recognizer/raspberry_pi
```

Запустить сервер:

```
hypercorn main:app --bind 0.0.0.0:8888
```

Запустить нужные для него севрисы:

```
docker run -p  8002:8002 --rm --device=/dev/video0:/dev/video0 --privileged stream:1.0
```

```
docker run -it --rm -p  8000:8000 -v "/etc/alsa:/etc/alsa" -v "/usr/share/alsa:/usr/share/alsa" -v "/home/ichiro/.config/pulse:/.config/pulse" -v "/run/user/$UID/pulse/native:/run/user/$UID/pulse/native" --env "PULSE_SERVER=unix:/run/user/$UID/pulse/native" --user "$(id -u)" patrik:1.0
```

Пример триггера на сервере
```
{
  "name": "thumb_up",
  "trigger_type": 2,
  "busy": true,
  "phrase": "string",
  "face_encoding": "string",
  "gesture_landmarks": "Thumb_Up",
  "startup": true,
  "week": 0,
  "period": 1,
  "number": 0,
  "face_rule": "0"
}
```