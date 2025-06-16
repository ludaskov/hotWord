import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import logging
from torch.utils.data import DataLoader, random_split
from dataset import KeywordDataset

# Настройка логгера
logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

class CNNKeywordSpotter(nn.Module):
    """
    Нейронная сеть для обнаружения ключевого слова на основе мел-спектрограмм.
    Использует сверточные слои для извлечения признаков и полносвязный слой для классификации.
    """

    def __init__(self):
        """
        Инициализация модели: сверточные слои для извлечения признаков и классификатор.
        """

        super().__init__()

        # Сверточные слои для извлечения признаков из мел-спектрограммы
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 32, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 32, H/2, W/2]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, H/2, W/2]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 64, H/4, W/4]

            nn.Dropout(0.3)                              # Dropout для борьбы с переобучением
        )

        # Классификатор, преобразующий признаки в 2 класса: 'not_stones', 'stones'
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Усреднение до [B, 64, 1, 1]
            nn.Flatten(),                 # [B, 64]
            nn.Linear(64, 2)              # Два выхода — бинарная классификация
        )

    def forward(self, x):
        """
        Прямой проход через модель.

        :param x: Входной тензор (B x 1 x H x W) — батч мел-спектрограмм
        :return: Классы (логиты) — (B x 2)
        """

        x = self.features(x)    # Извлекаем признаки
        x = self.classifier(x)  # Классифицируем
        return x

    def train_model(self, train_loader, val_loader, epochs=20, lr=0.001, threshold=0.7, save_path="model.pth"):
        """
        Обучает модель на заданных данных.

        :param train_loader: DataLoader для обучающей выборки
        :param val_loader: DataLoader для валидационной выборки
        :param epochs: Количество эпох обучения
        :param lr: Скорость обучения
        :param threshold: Порог вероятности для класса "stones"
        :param save_path: Путь для сохранения модели после обучения
        """
        
        # Оптимизатор и функция потерь
        optimizer = Adam(self.parameters(), lr=lr)

        # Взвешивание классов: больше вес для класса "stones" (1), он встречается реже
        class_weights = torch.tensor([1.0, 4.0], dtype=torch.float32).to(next(self.parameters()).device)
        criterion = CrossEntropyLoss(weight=class_weights)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            # -------- ОБУЧЕНИЕ --------
            for mel, label in train_loader:
                if mel.dim() == 3:
                    mel = mel.unsqueeze(1)  # [B, 1, H, W] — добавляем размерность канала

                mel = mel.to(next(self.parameters()).device)
                label = label.to(next(self.parameters()).device)

                optimizer.zero_grad()
                output = self(mel)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            # -------- ВАЛИДАЦИЯ --------
            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            false_positives = 0
            false_negatives = 0

            with torch.no_grad():
                for mel, label in val_loader:
                    if mel.dim() == 3:
                        mel = mel.unsqueeze(1)

                    mel = mel.to(next(self.parameters()).device)
                    label = label.to(next(self.parameters()).device)

                    output = self(mel)
                    loss = criterion(output, label)
                    val_loss += loss.item()

                    # Softmax и пороговое принятие решения
                    probs = torch.softmax(output, dim=1)
                    predicted = (probs[:, 1] > threshold).long()  # Предсказание "stones" если prob > threshold

                    # Подсчет метрик
                    correct += (predicted == label).sum().item()
                    total += label.size(0)

                    false_positives += ((predicted == 1) & (label == 0)).sum().item()
                    false_negatives += ((predicted == 0) & (label == 1)).sum().item()

            val_loss /= len(val_loader)
            val_acc = correct / total

            # Вывод в консоль и лог-файл
            log_msg = (
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
                f"Val Acc={val_acc:.4f} | FP={false_positives} | FN={false_negatives}"
            )
            print(log_msg)
            logging.info(log_msg)

        # Сохраняем модель
        torch.save(self.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")
        print(f"Model saved to {save_path}")



# =======================
# Запуск обучения
# =======================
if __name__ == "__main__":
    dataset = KeywordDataset("data") # Загружаем датасет
    val_size = int(0.2 * len(dataset)) # Определяем размер валидационной выборки
    train_size = len(dataset) - val_size # Определяем размер тренировочной выборки

    # Случайным образом делим датасет на тренировочную и валидационную части
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Создаём DataLoader'ы для загрузки данных батчами
    # shuffle=True перемешивает тренировочные данные
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = CNNKeywordSpotter()
    model.train_model(train_loader, val_loader, epochs=15)
