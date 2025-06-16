import threading
import time
from collections import deque
from pydub import AudioSegment
from io import BytesIO
import requests
import os
import logging

import torch
import torch.nn.functional as F
import numpy as np

from model import CNNKeywordSpotter
from dataset import KeywordDataset

# ==== Логирование ====
logging.basicConfig(
    filename="logs/logs.log",
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==== Настройки ====
RADIO_URL = "https://radio.kotah.ru/thanosshow" # thanosshow soundcheck
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
    """
    Обрезает или дополняет мел-спектрограмму до нужной ширины.
    :param mel: Мел-спектрограмма [B, 1, H, W] или [1, H, W]
    :param target_width: Целевая ширина
    :return: Приведённая по размеру мел-спектрограмма
    """

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
    """Читает поток MP3 и записывает в байтовый буфер."""

    global buffer
    while True:
        try:
            logger.info("Подключение к радио...")
            print("Подключение к радио...")
            response = requests.get(RADIO_URL, stream=True, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Ошибка подключения: статус {response.status_code}")
                print(f"Ошибка подключения: статус {response.status_code}")
                time.sleep(1)
                continue

            logger.info("Поток подключён. Чтение данных...")
            print("Поток подключён. Чтение данных...")
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    with buffer_lock:
                        buffer.write(chunk)
                time.sleep(0.01)

        except requests.exceptions.RequestException as e:
            logger.warning("Нет соединения с радио. Повтор через 1 сек...")
            print("Нет соединения с радио. Повтор через 1 сек...")
            time.sleep(1)

def audio_processor():
    """
    Обрабатывает байтовый буфер и преобразует данные в AudioSegment.
    Сохраняет последние BUFFER_SECONDS секунд в deque.
    """

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
                logger.warning(f"Ошибка чтения MP3: {e}")
                print(f"Ошибка чтения MP3: {e}")
        time.sleep(0.5)

def detect_and_save():
    """
    Обрабатывает последние WINDOW_DURATION секунд звука.
    Пропускает их через модель, и если вероятность класса "stones"
    выше порога, сохраняет фрагмент POST_RECORD_DURATION секунд.
    """
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

        # Вычисляем мел-спектрограмму
        mel = mel_spec(waveform)
        if mel.dim() == 3:
            mel = mel.unsqueeze(0)
        mel = pad_or_trim(mel)

        # Предсказание модели
        with torch.no_grad():
            output = model(mel)
            probs = torch.softmax(output, dim=1)
            prob = probs[0, 0].item()

        if prob > THRESHOLD:
            logger.info(f"Найдено 'stones'! (prob={prob:.2f})")
            print(f"Найдено 'stones'! (prob={prob:.2f})")
            # Сохраняем последние POST_RECORD_DURATION секунд
            with buffer_lock:
                combined = sum(audio_buffer, AudioSegment.empty())
                if combined.duration_seconds >= POST_RECORD_DURATION:
                    segment = combined[-int(POST_RECORD_DURATION * 1000):]
                    os.makedirs("online_detected_stones", exist_ok=True)
                    filename = f"online_detected_stones/stones_{save_counter:04d}.wav"
                    segment.export(filename, format="wav")
                    logger.info(f"Сохранено: {filename}")
                    print(f"Сохранено: {filename}")
                    save_counter += 1
            time.sleep(2.5)  # Пауза, чтобы не ловить один и тот же фрагмент повторно
        else:
            logger.info("Слушаем...")
            print(f"Слушаем...")




# =======================
# Запуск поиска
# =======================
if __name__ == "__main__":
    logger.info("Запуск...")
    print("Запуск...")
    threading.Thread(target=stream_reader, daemon=True).start()
    threading.Thread(target=audio_processor, daemon=True).start()
    detect_and_save()
