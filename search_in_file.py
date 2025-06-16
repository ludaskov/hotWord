import torch
import torch.nn.functional as F
import torchaudio
from model import CNNKeywordSpotter
from dataset import KeywordDataset
import os
import logging

# ==== Логирование ====
logging.basicConfig(
    filename="logs/logs.log",
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def pad_or_trim(mel, target_width=78):
    """
    Обрезает или дополняет мел-спектрограмму до нужной ширины.
    :param mel: Мел-спектрограмма [B, 1, H, W] или [1, H, W]
    :param target_width: Целевая ширина
    :return: Приведённая по размеру мел-спектрограмма
    """

    # mel shape может быть (batch, channels, height, width)
    if mel.dim() == 4:
        batch, channels, h, w = mel.shape
        if w < target_width:
            pad_amount = target_width - w
            mel = F.pad(mel, (0, pad_amount))  # padding справа по ширине
        elif w > target_width:
            mel = mel[:, :, :, :target_width]  # обрезаем справа
    elif mel.dim() == 3:
        channels, h, w = mel.shape
        if w < target_width:
            pad_amount = target_width - w
            mel = F.pad(mel, (0, pad_amount))
        elif w > target_width:
            mel = mel[:, :, :target_width]
    else:
        raise ValueError(f"Unexpected mel dimensions: {mel.shape}")
    return mel

def sliding_predict(audio_path, model_path="model.pth", sample_rate=16000):
    """
    Скользящим окном по аудиофайлу определяет наличие ключевого слова 'stones'.
    :param audio_path (str): Путь до входного WAV-файла.
    :param model_path (str): Путь к сохранённой модели PyTorch.
    :param sample_rate (int): Частота дискретизации для модели.
    :return: Список кортежей (timestamp, label), где label — 'stones' или 'not_stones'.
    """

    # Загрузка и ресемпл аудио
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resample = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resample(waveform)
    waveform = waveform[0]  # берем моно канал (1D тензор)

    window_size = sample_rate       # 1 секунда в сэмплах
    step_size = sample_rate #// 2   # шаг 1 (0.5) секунда

    # Инициализация модели и мел-спектрограммы
    model = CNNKeywordSpotter()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    mel_spec = KeywordDataset("data").mel_spec

    results = []     # Результаты классификации по каждому окну
    save_counter = 0 # Счётчик сохранений сегментов

    for start in range(0, waveform.size(0) - window_size + 1, step_size):
        end = start + window_size
        chunk = waveform[start:end].unsqueeze(0)  # (1, window_size)

        mel = mel_spec(chunk)  # возвращает (1, 64, W)

        if mel.dim() == 3:
            mel = mel.unsqueeze(0)  # (1, 1, 64, W)

        mel = pad_or_trim(mel, 78)  # (1, 1, 64, 78)
        assert mel.dim() == 4

        threshold = 0.9 # порог уверенности модели

        # Предсказание модели
        with torch.no_grad():
            output = model(mel)
            probs = torch.softmax(output, dim=1)
            stones_prob = probs[0, 0].item()

        if stones_prob > threshold:
            label = "stones"
        else:
            label = "not_stones"

        timestamp = round(start / sample_rate, 2)
        if label == "stones":
            logger.info("Найдено 'stones'")
            print(f"Найдено 'stones' на {timestamp:.2f}сек (prob={stones_prob:.2f})")

            save_start = end
            save_end = save_start + int(2 * sample_rate)  # 2 секунды в сэмплах
            if save_end > waveform.size(0):
                save_end = waveform.size(0)

            segment = waveform[save_start:save_end]
            if segment.numel() > 0:
                os.makedirs("detected_stones", exist_ok=True)
                save_path = os.path.join("detected_stones", f"stones_{save_counter}_{timestamp:.2f}s.wav")
                torchaudio.save(save_path, segment.unsqueeze(0), sample_rate)
                logger.info("Сохранено")
                print(f"Сохранено {save_path}")
                save_counter += 1
        results.append((timestamp, label))

    return results



# =======================
# Запуск поиска
# =======================
if __name__ == "__main__":
    results = sliding_predict("data/input/thanos_message.wav")
    