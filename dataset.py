import os
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram

class KeywordDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, n_mels=64):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.mel_spec = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.files = []
        self.labels = []

        for label, folder in enumerate(["stones", "not_stones"]):
            folder_path = os.path.join(root_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.endswith(".wav"):
                    self.files.append(os.path.join(folder_path, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])  # waveform: [channels, samples]

        # Приведение к моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # [1, samples]

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Обрезка/дополнение до 1 секунды
        target_len = self.sample_rate
        if waveform.shape[1] < target_len:
            pad_len = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        elif waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]

        # Мел-спектрограмма
        mel = self.mel_spec(waveform)  # [1, n_mels, time]

        # Обеспечиваем фиксированную длину по времени
        target_frames = 81
        if mel.shape[2] < target_frames:
            mel = torch.nn.functional.pad(mel, (0, target_frames - mel.shape[2]))
        elif mel.shape[2] > target_frames:
            mel = mel[:, :, :target_frames]

        # mel: [1, 64, 81]
        return mel.squeeze(0), torch.tensor(self.labels[idx])  # -> [64, 81]