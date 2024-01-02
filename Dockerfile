# docker run -it --rm --network host gestures:1.0

FROM python:3.11-slim

WORKDIR /gestures
COPY . .
RUN    pip install pip --upgrade \
    # && pip install mediapipe \
    # && pip install requests \
    # && pip install websockets \
    # && pip install fastapi \
    # && pip install hypercorn \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall opencv-contrib-python -y \
    && pip install opencv-python-headless
CMD [ "hypercorn", "main:app", "--bind", "0.0.0.0:8005" ]