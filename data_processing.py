import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.feature_info = {}

    def load_data(self, file_path):
        """
        Загрузка данных из CSV файла
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Данные загружены успешно. Размер: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            return None

    def validate_data(self, data):
        """
        Валидация структуры данных
        """
        # Убираем inbnd_type из обязательных колонок
        required_columns = [
            'task_id', 'temp_zone', 'sku_count',
            'accepted_quantity', 'accepted_weight_kg', 'accepted_volume_litr',
            'inbnd_duration_min'
        ]

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Отсутствуют обязательные колонки: {missing_columns}")
            return False

        # Проверка на пропущенные значения
        if data.isnull().sum().sum() > 0:
            logger.warning("Обнаружены пропущенные значения")

        logger.info("Валидация данных прошла успешно")
        return True

    def handle_missing_target(self, data, target_column='inbnd_duration_min'):
        """
        Обработка пропущенных значений в целевой колонке
        """
        missing_count = data[target_column].isna().sum()

        if missing_count > 0:
            logger.warning(f"Обнаружено {missing_count} пропущенных значений в целевой колонке")

            # Удаление строк с пропущенными значениями (рекомендуется)
            data_cleaned = data.dropna(subset=[target_column])
            logger.info(f"Удалено {missing_count} строк с пропущенными значениями")
            return data_cleaned

        return data

    def filter_by_temp_zone(self, data, temp_zone):
        """
        Фильтрация данных по температурной зоне
        """
        filtered_data = data[data['temp_zone'] == temp_zone].copy()
        logger.info(f"Данные отфильтрованы для зоны {temp_zone}. Размер: {filtered_data.shape}")
        return filtered_data

    def preprocess_data(self, data, temp_zone=None):
        """
        Предварительная обработка данных для конкретной температурной зоны
        """
        # Создание копии данных
        processed_data = data.copy()

        # Удаление task_id из признаков (не нужен для предсказания)
        if 'task_id' in processed_data.columns:
            processed_data = processed_data.drop('task_id', axis=1)

        # Удаление inbnd_type из признаков (согласно требованию)
        if 'inbnd_type' in processed_data.columns:
            processed_data = processed_data.drop('inbnd_type', axis=1)

        # Если указана температурная зона, фильтруем данные
        if temp_zone:
            processed_data = self.filter_by_temp_zone(processed_data, temp_zone)

        # Обработка пропущенных значений в целевой колонке
        processed_data = self.handle_missing_target(processed_data)

        # Удаление temp_zone из признаков (так как теперь мы обучаем отдельные модели)
        if 'temp_zone' in processed_data.columns:
            processed_data = processed_data.drop('temp_zone', axis=1)

        # Обработка числовых признаков
        numerical_features = ['sku_count', 'accepted_quantity', 'accepted_weight_kg', 'accepted_volume_litr']

        for feature in numerical_features:
            if feature in processed_data.columns:
                # Заполнение пропущенных значений медианой
                processed_data[feature] = processed_data[feature].fillna(processed_data[feature].median())

                # Сохраняем информацию о диапазонах для веб-интерфейса
                self.feature_info[feature] = {
                    'min': float(processed_data[feature].min()),
                    'max': float(processed_data[feature].max()),
                    'mean': float(processed_data[feature].mean())
                }

        logger.info(f"Предобработка данных завершена для зоны {temp_zone}")
        return processed_data

    def get_feature_info(self):
        """
        Получение информации о признаках для веб-интерфейса
        """
        return self.feature_info

    def get_temp_zones(self, data):
        """
        Получение списка уникальных температурных зон
        """
        if 'temp_zone' in data.columns:
            return list(data['temp_zone'].unique())
        return []
