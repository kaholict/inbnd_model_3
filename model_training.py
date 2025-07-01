import pandas as pd
import numpy as np
from pycaret.regression import *
import pickle
import os
import logging
from data_processing import DataProcessor

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {}  # Словарь для хранения моделей по температурным зонам
        self.setup_ml = None

    def setup_environment(self, data, target_column='inbnd_duration_min'):
        """
        Настройка окружения PyCaret для регрессии
        """
        try:
            # Настройка PyCaret для задачи регрессии
            self.setup_ml = setup(
                data=data,
                target=target_column,
                train_size=0.8,
                session_id=123,
                verbose=False
            )

            logger.info("Окружение PyCaret настроено успешно")
            return True
        except Exception as e:
            logger.error(f"Ошибка при настройке окружения: {e}")
            return False

    def compare_models(self):
        """
        Сравнение различных алгоритмов машинного обучения
        """
        try:
            # Сравнение моделей (автоматически тестирует множество алгоритмов)
            best_models = compare_models(
                include=['lr', 'rf', 'gbr', 'xgboost', 'catboost',
                         'ridge', 'lasso', 'en', 'dt', 'ada'],
                sort='MAE',  # Сортировка по средней абсолютной ошибке
                n_select=3,  # Выбираем топ-3 модели
                verbose=False
            )

            logger.info("Сравнение моделей завершено")
            return best_models
        except Exception as e:
            logger.error(f"Ошибка при сравнении моделей: {e}")
            return None

    def tune_hyperparameters(self, model):
        """
        Настройка гиперпараметров для лучшей модели
        """
        try:
            tuned_model = tune_model(
                model,
                optimize='MAE',
                search_library='scikit-learn',
                verbose=False
            )

            logger.info("Настройка гиперпараметров завершена")
            return tuned_model
        except Exception as e:
            logger.error(f"Ошибка при настройке гиперпараметров: {e}")
            return model

    def train_model_for_temp_zone(self, data, temp_zone):
        """
        Обучение модели для конкретной температурной зоны
        """
        logger.info(f"Начинаем обучение модели для зоны: {temp_zone}")

        # Предобработка данных для конкретной температурной зоны
        processed_data = self.data_processor.preprocess_data(data, temp_zone)

        if processed_data.empty:
            logger.warning(f"Нет данных для температурной зоны {temp_zone}")
            return False

        # Настройка окружения
        if not self.setup_environment(processed_data):
            return False

        # Сравнение моделей
        best_models = self.compare_models()
        if best_models is None:
            return False

        # Выбор лучшей модели
        if isinstance(best_models, list):
            best_model = best_models[0]
        else:
            best_model = best_models

        # Настройка гиперпараметров
        tuned_model = self.tune_hyperparameters(best_model)

        # Финализация модели (обучение на полном датасете)
        final_model = finalize_model(tuned_model)

        # Сохранение модели для данной температурной зоны
        self.models[temp_zone] = final_model

        logger.info(f"Обучение модели для зоны {temp_zone} завершено успешно")
        return True

    def train_all_models(self, data):
        """
        Обучение моделей для всех температурных зон
        """
        # Получение списка уникальных температурных зон
        temp_zones = self.data_processor.get_temp_zones(data)

        if not temp_zones:
            logger.error("Не найдены температурные зоны в данных")
            return False

        success_count = 0
        for temp_zone in temp_zones:
            if self.train_model_for_temp_zone(data, temp_zone):
                success_count += 1

        logger.info(f"Обучено {success_count} из {len(temp_zones)} моделей")
        return success_count > 0

    def save_models(self, filename_prefix='temp_zone_model'):
        """
        Сохранение всех обученных моделей
        """
        if not self.models:
            logger.error("Нет обученных моделей для сохранения")
            return False

        try:
            # Создание директории models, если её нет
            os.makedirs('models', exist_ok=True)

            # Сохранение каждой модели отдельно
            for temp_zone, model in self.models.items():
                # Очистка названия зоны для имени файла
                safe_zone_name = temp_zone.replace(' ', '_').replace('/', '_')
                filename = f'{filename_prefix}_{safe_zone_name}'

                # Сохранение модели с помощью PyCaret
                save_model(model, f'models/{filename}')

                logger.info(f"Модель для зоны {temp_zone} сохранена как {filename}")

            # Сохранение дополнительной информации
            model_info = {
                'data_processor': self.data_processor,
                'feature_info': self.data_processor.get_feature_info(),
                'temp_zones': list(self.models.keys())
            }

            with open(f'models/{filename_prefix}_info.pkl', 'wb') as f:
                pickle.dump(model_info, f)

            logger.info("Все модели сохранены успешно")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении моделей: {e}")
            return False

    def load_models(self, filename_prefix='temp_zone_model'):
        """
        Загрузка всех сохранённых моделей
        """
        try:
            # Загрузка дополнительной информации
            with open(f'models/{filename_prefix}_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
                self.data_processor = model_info['data_processor']
                temp_zones = model_info['temp_zones']

            # Загрузка каждой модели
            for temp_zone in temp_zones:
                safe_zone_name = temp_zone.replace(' ', '_').replace('/', '_')
                filename = f'{filename_prefix}_{safe_zone_name}'

                try:
                    model = load_model(f'models/{filename}')
                    self.models[temp_zone] = model
                    logger.info(f"Модель для зоны {temp_zone} загружена успешно")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке модели для зоны {temp_zone}: {e}")

            logger.info(f"Загружено {len(self.models)} моделей")
            return len(self.models) > 0
        except Exception as e:
            logger.error(f"Ошибка при загрузке моделей: {e}")
            return False

    def predict_for_temp_zone(self, input_data, temp_zone):
        """
        Предсказание для конкретной температурной зоны
        """
        if temp_zone not in self.models:
            logger.error(f"Модель для зоны {temp_zone} не найдена")
            return None

        try:
            # Подготовка данных для предсказания (убираем temp_zone из входных данных)
            processed_input = input_data.copy()
            if 'temp_zone' in processed_input.columns:
                processed_input = processed_input.drop('temp_zone', axis=1)

            # Предсказание
            prediction = predict_model(self.models[temp_zone], data=processed_input, verbose=False)

            return prediction['prediction_label'].iloc[0] if 'prediction_label' in prediction.columns else \
            prediction.iloc[0, -1]
        except Exception as e:
            logger.error(f"Ошибка при предсказании для зоны {temp_zone}: {e}")
            return None

    def get_available_temp_zones(self):
        """
        Получение списка доступных температурных зон
        """
        return list(self.models.keys())


def main():
    """
    Основная функция для обучения моделей
    """
    # Инициализация обработчика данных
    data_processor = DataProcessor()

    # Загрузка данных
    data = data_processor.load_data('dataset.csv')

    if data is None:
        logger.error("Не удалось загрузить данные")
        return

    # Валидация данных
    if not data_processor.validate_data(data):
        logger.error("Данные не прошли валидацию")
        return

    # Инициализация и обучение моделей
    trainer = ModelTrainer(data_processor)

    if trainer.train_all_models(data):
        # Сохранение моделей
        trainer.save_models()
        logger.info("Обучение всех моделей завершено успешно!")
    else:
        logger.error("Ошибка при обучении моделей")


if __name__ == "__main__":
    main()
