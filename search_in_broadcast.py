import threading
import time
from collections import deque
from pydub import AudioSegment
from io import BytesIO
import requests
import os

import torch
import torch.nn.functional as F
import numpy as np

from model import CNNKeywordSpotter
from dataset import KeywordDataset

# ==== Настройки ====
RADIO_URL = "https://radio.kotah.ru/soundcheck" # thanosshow
CHUNK_SIZE = 1024
BUFFER_SECONDS = 5
SAMPLE_RATE = 16000  # для модели
THRESHOLD = 0.9
WINDOW_DURATION = 1  # сек
POST_RECORD_DURATION = 2  # сек

buffer = BytesIO()
buffer_lock = threading.Lock()
audio_buffer = deque()  # Хранит AudioSegment фрагменты

# ==== Модель ====
model = CNNKeywordSpotter()
model.load_state_dict(torch.load("model.pth"))
model.eval()

mel_spec = KeywordDataset("data").mel_spec

def pad_or_trim(mel, target_width=78):
    if mel.dim() == 4:
        _, _, _, w = mel.shape
        if w < target_width:
            mel = F.pad(mel, (0, target_width - w))
        else:
            mel = mel[:, :, :, :target_width]
    elif mel.dim() == 3:
        _, _, w = mel.shape
        if w < target_width:
            mel = F.pad(mel, (0, target_width - w))
        else:
            mel = mel[:, :, :target_width]
    return mel

def stream_reader():
    global buffer
    while True:
        try:
            print("Подключение к радио...")
            response = requests.get(RADIO_URL, stream=True, timeout=10)
            if response.status_code != 200:
                print(f"Ошибка подключения: статус {response.status_code}")
                time.sleep(1)
                continue

            print("Подключено. Чтение потока...")
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    with buffer_lock:
                        buffer.write(chunk)
                time.sleep(0.01)

        except requests.exceptions.RequestException as e:
            print("Нет соединения с радио. Повтор через 1 сек...")
            time.sleep(1)

def audio_processor():
    global buffer, audio_buffer
    while True:
        with buffer_lock:
            data = buffer.getvalue()
            buffer = BytesIO()
        if data:
            try:
                segment = AudioSegment.from_mp3(BytesIO(data))
                segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
                audio_buffer.append(segment)
                # Ограничим размер буфера
                total_duration = sum(s.duration_seconds for s in audio_buffer)
                while total_duration > BUFFER_SECONDS:
                    removed = audio_buffer.popleft()
                    total_duration -= removed.duration_seconds
            except Exception as e:
                print(f"Ошибка чтения MP3: {e}")
        time.sleep(0.5)

def detect_and_save():
    save_counter = 0
    while True:
        time.sleep(0.5)

        with buffer_lock:
            if not audio_buffer:
                continue
            combined = sum(audio_buffer, AudioSegment.empty())

        if combined.duration_seconds < WINDOW_DURATION:
            continue

        # Получаем последние WINDOW_DURATION секунд
        window = combined[-int(WINDOW_DURATION * 1000):]
        samples = np.array(window.get_array_of_samples()).astype(np.float32) / 32768.0
        waveform = torch.tensor(samples).unsqueeze(0)

        if waveform.size(1) < SAMPLE_RATE:
            continue

        waveform = waveform[:, :SAMPLE_RATE]

        mel = mel_spec(waveform)
        if mel.dim() == 3:
            mel = mel.unsqueeze(0)
        mel = pad_or_trim(mel)

        with torch.no_grad():
            output = model(mel)
            probs = torch.softmax(output, dim=1)
            prob = probs[0, 0].item()

        if prob > THRESHOLD:
            print(f"Найдено 'stones'! (prob={prob:.2f})")
            # Сохраняем последние POST_RECORD_DURATION секунд
            with buffer_lock:
                combined = sum(audio_buffer, AudioSegment.empty())
                if combined.duration_seconds >= POST_RECORD_DURATION:
                    segment = combined[-int(POST_RECORD_DURATION * 1000):]
                    os.makedirs("online_detected_stones", exist_ok=True)
                    filename = f"online_detected_stones/stones_{save_counter:04d}.wav"
                    segment.export(filename, format="wav")
                    print(f"Сохранено: {filename}")
                    save_counter += 1
            time.sleep(2.5)  # чтобы не сохранять каждый кадр подряд
        else:
            print(f"Слушаем...")

if __name__ == "__main__":
    print("Запуск...")
    threading.Thread(target=stream_reader, daemon=True).start()
    threading.Thread(target=audio_processor, daemon=True).start()
    detect_and_save()
