import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model_training import ModelTrainer
from data_processing import DataProcessor
import os
import logging

# Настройка страницы
st.set_page_config(
    page_title="Планирование приёмки по температурным зонам",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_trained_models():
    """
    Загрузка обученных моделей для всех температурных зон (кэшируется для ускорения)
    """
    data_processor = DataProcessor()
    trainer = ModelTrainer(data_processor)

    if os.path.exists('models/temp_zone_model_info.pkl'):
        if trainer.load_models():
            return trainer, data_processor
        else:
            return None, None
    else:
        return None, None


def create_default_feature_info():
    """
    Создание информации о признаках по умолчанию для каждой температурной зоны отдельно
    """
    try:
        df = pd.read_csv('dataset.csv')

        # Создание словаря для каждой температурной зоны
        feature_info_by_zone = {}
        temp_zones = ['Сухой', 'Холод', 'Заморозка']

        for zone in temp_zones:
            # Фильтрация данных по температурной зоне
            zone_data = df[df['temp_zone'] == zone]

            if not zone_data.empty:
                feature_info_by_zone[zone] = {
                    'sku_count': {
                        'min': int(zone_data['sku_count'].min()),
                        'max': int(zone_data['sku_count'].max()),
                        'mean': round(zone_data['sku_count'].mean(), 0)
                    },
                    'accepted_quantity': {
                        'min': int(zone_data['accepted_quantity'].min()),
                        'max': int(zone_data['accepted_quantity'].max()),
                        'mean': round(zone_data['accepted_quantity'].mean(), 0)
                    },
                    'accepted_weight_kg': {
                        'min': round(zone_data['accepted_weight_kg'].min(), 2),
                        'max': round(zone_data['accepted_weight_kg'].max(), 2),
                        'mean': round(zone_data['accepted_weight_kg'].mean(), 2)
                    },
                    'accepted_volume_litr': {
                        'min': round(zone_data['accepted_volume_litr'].min(), 2),
                        'max': round(zone_data['accepted_volume_litr'].max(), 2),
                        'mean': round(zone_data['accepted_volume_litr'].mean(), 2)
                    }
                }
            else:
                # Если нет данных для зоны, используем общие значения по умолчанию
                feature_info_by_zone[zone] = {
                    'sku_count': {'min': 1, 'max': 100, 'mean': 20},
                    'accepted_quantity': {'min': 10, 'max': 1000, 'mean': 100},
                    'accepted_weight_kg': {'min': 0.1, 'max': 50.0, 'mean': 10.0},
                    'accepted_volume_litr': {'min': 0.1, 'max': 20.0, 'mean': 5.0}
                }

        return feature_info_by_zone

    except Exception as e:
        logger.warning(f"Ошибка при загрузке данных для расчета значений по умолчанию: {e}")
        # Возвращаем значения по умолчанию для всех зон
        default_values = {
            'sku_count': {'min': 1, 'max': 100, 'mean': 20},
            'accepted_quantity': {'min': 10, 'max': 1000, 'mean': 100},
            'accepted_weight_kg': {'min': 0.1, 'max': 50.0, 'mean': 10.0},
            'accepted_volume_litr': {'min': 0.1, 'max': 20.0, 'mean': 5.0}
        }

        return {zone: default_values for zone in ['Сухой', 'Холод', 'Заморозка']}


def create_temp_zone_inputs():
    """
    Создание табличного ввода для всех температурных зон с индивидуальными значениями по умолчанию
    """
    st.header("📊 Параметры приёмки по температурным зонам")

    # Определение температурных зон
    temp_zones = ['Сухой', 'Холод', 'Заморозка']

    # Создание таблицы для ввода данных
    st.subheader("Средние показатели по зонам")

    # Создание колонок для таблицы
    cols = st.columns([2] + [1.5] * len(temp_zones))

    # Заголовки таблицы
    cols[0].write("**Параметр**")
    for i, zone in enumerate(temp_zones):
        cols[i + 1].write(f"**{zone}**")

    # Получение информации о признаках для каждой зоны отдельно
    feature_info_by_zone = create_default_feature_info()

    # Словарь для хранения всех входных данных
    zone_inputs = {}

    # Инициализация словарей для каждой зоны
    for zone in temp_zones:
        zone_inputs[zone] = {}

    # Создание строк для каждого параметра
    parameters = [
        ('sku_count', 'Количество SKU', 'шт'),
        ('accepted_quantity', 'Принятое количество', 'шт'),
        ('accepted_weight_kg', 'Вес приёмки', 'кг'),
        ('accepted_volume_litr', 'Объём приёмки', 'л')
    ]

    for param_key, param_name, unit in parameters:
        cols = st.columns([2] + [1.5] * len(temp_zones))
        cols[0].write(f"{param_name} ({unit}):")

        for i, zone in enumerate(temp_zones):
            # Получение значения по умолчанию для конкретной зоны
            default_value = feature_info_by_zone[zone][param_key]['mean']
            min_value = feature_info_by_zone[zone][param_key]['min']
            max_value = feature_info_by_zone[zone][param_key]['max']

            # Создание элемента ввода с индивидуальными параметрами для каждой зоны
            if param_key in ['accepted_weight_kg', 'accepted_volume_litr']:
                zone_inputs[zone][param_key] = cols[i + 1].number_input(
                    f"{param_name}_{zone}",
                    value=float(default_value),
                    min_value=float(min_value),
                    max_value=float(max_value),
                    step=0.1,
                    format="% .2f",
                    key=f"{param_key}_{zone}",
                    label_visibility="collapsed",
                    help=f"Среднее значение для {zone}: {default_value:_.2f} {unit}".replace('_', ' ')
                )
            else:
                zone_inputs[zone][param_key] = cols[i + 1].number_input(
                    f"{param_name}_{zone}",
                    value=int(default_value),
                    min_value=int(min_value),
                    max_value=int(max_value),
                    step=1,
                    key=f"{param_key}_{zone}",
                    label_visibility="collapsed",
                    help=f"Среднее значение для {zone}: {int(default_value)} {unit}"
                )

    # Отображение статистики по зонам в развернутом блоке
    with st.expander("📈 Статистика по температурным зонам"):
        st.subheader("Средние значения параметров по зонам")

        # Создание таблицы со статистикой
        stats_data = []
        for zone in temp_zones:
            stats_data.append({
                'Температурная зона': zone,
                'Среднее количество SKU': int(feature_info_by_zone[zone]['sku_count']['mean']),
                'Среднее принятое количество': int(feature_info_by_zone[zone]['accepted_quantity']['mean']),
                'Средний вес (кг)': feature_info_by_zone[zone]['accepted_weight_kg']['mean'],
                'Средний объём (л)': feature_info_by_zone[zone]['accepted_volume_litr']['mean']
            })

        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

        # Создание графика сравнения средних значений
        col1, col2 = st.columns(2)

        with col1:
            # График средних весов по зонам
            fig_weight = px.bar(
                x=temp_zones,
                y=[feature_info_by_zone[zone]['accepted_weight_kg']['mean'] for zone in temp_zones],
                title="Средний вес приёмки по зонам",
                labels={'x': 'Температурная зона', 'y': 'Вес (кг)'},
                color=temp_zones
            )
            fig_weight.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_weight, use_container_width=True)

        with col2:
            # График средних объёмов по зонам
            fig_volume = px.bar(
                x=temp_zones,
                y=[feature_info_by_zone[zone]['accepted_volume_litr']['mean'] for zone in temp_zones],
                title="Средний объём приёмки по зонам",
                labels={'x': 'Температурная зона', 'y': 'Объём (л)'},
                color=temp_zones
            )
            fig_volume.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_volume, use_container_width=True)

    return zone_inputs, temp_zones


def create_additional_inputs():
    """
    Создание дополнительных параметров ввода
    """
    st.header("⚙️ Дополнительные параметры склада")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Количество ворот по зонам")
        gates = {}
        gates['Сухой'] = st.number_input(
            "Количество ворот - Сухой:",
            value=5,
            step=1,
            help="Количество доступных ворот в сухой зоне"
        )
        gates['Холод'] = st.number_input(
            "Количество ворот - Холод:",
            value=3,
            step=1,
            help="Количество доступных ворот в холодной зоне"
        )
        gates['Заморозка'] = st.number_input(
            "Количество ворот - Заморозка:",
            value=2,
            step=1,
            help="Количество доступных ворот в зоне заморозки"
        )

    with col2:
        st.subheader("Временные параметры")
        unload_time = st.number_input(
            "Среднее время разгрузки одной ТС (мин):",
            value=0.0,
            step=1.0,
            format="% .1f",
            help="Время, необходимое для разгрузки одного транспортного средства"
        )

        shift_hours = st.number_input(
            "Количество часов в смене, доступных на приёмку:",
            value=8.0,
            step=0.5,
            format="% .1f",
            help="Количество рабочих часов в смене для приёмки товаров"
        )

    return gates, unload_time, shift_hours


def create_demo_prediction(inputs, temp_zone):
    """
    Демонстрационное предсказание на основе простой формулы
    """
    # Простая формула для демонстрации с учетом температурной зоны
    base_prediction = (
            inputs['sku_count'] * 0.5 +
            inputs['accepted_quantity'] * 0.1 +
            inputs['accepted_weight_kg'] * 2 +
            inputs['accepted_volume_litr'] * 1.5
    )

    # Корректировка по температурной зоне
    zone_multipliers = {
        'Сухой': 1.0,
        'Холод': 1.2,
        'Заморозка': 1.5
    }

    prediction = base_prediction * zone_multipliers.get(temp_zone, 1.0)
    prediction += np.random.normal(0, 2)  # Добавляем небольшой шум

    # Убеждаемся, что предсказание положительное
    prediction = max(5, prediction)

    return prediction


def calculate_capacity_metrics(avg_times, gates, unload_time, shift_hours, zone_inputs):
    """
    Расчет метрик производительности склада
    """
    results = {}

    for zone in avg_times.keys():
        # Расчет количества приёмок
        total_time_per_reception = unload_time + avg_times[zone]
        if total_time_per_reception > 0:
            receptions_per_shift = (gates[zone] * shift_hours * 60) / total_time_per_reception
        else:
            receptions_per_shift = 0

        # Расчет общего принятого веса
        total_weight_per_shift = (zone_inputs[zone]['accepted_weight_kg'] *
                                  gates[
                                      zone] * shift_hours * 60) / total_time_per_reception if total_time_per_reception > 0 else 0

        results[zone] = {
            'avg_time': avg_times[zone],
            'receptions_per_shift': receptions_per_shift,
            'total_weight_per_shift': total_weight_per_shift
        }

    return results


def create_results_visualization(results):
    """
    Создание визуализации результатов
    """
    st.header("📈 Результаты расчетов")

    # Создание таблицы результатов
    st.subheader("Сводная таблица по температурным зонам")

    results_data = []
    for zone, metrics in results.items():
        results_data.append({
            'Температурная зона': zone,
            'Среднее время приёмки (мин)': f"{metrics['avg_time']:_.1f}".replace('_', ' '),
            'Количество приёмок за смену': f"{metrics['receptions_per_shift']:_.1f}".replace('_', ' '),
            'Общий вес за смену (кг)': f"{metrics['total_weight_per_shift']:_.1f}".replace('_', ' ')
        })

    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # Создание графиков
    col1, col2 = st.columns(2)

    with col1:
        # График среднего времени приёмки
        fig_time = px.bar(
            x=list(results.keys()),
            y=[metrics['avg_time'] for metrics in results.values()],
            title="Среднее время приёмки по зонам",
            labels={'x': 'Температурная зона', 'y': 'Время (мин)'},
            color=list(results.keys())
        )
        fig_time.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        # График количества приёмок за смену
        fig_capacity = px.bar(
            x=list(results.keys()),
            y=[metrics['receptions_per_shift'] for metrics in results.values()],
            title="Количество приёмок за смену",
            labels={'x': 'Температурная зона', 'y': 'Количество приёмок'},
            color=list(results.keys())
        )
        fig_capacity.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_capacity, use_container_width=True)

    # График общего веса
    fig_weight = px.bar(
        x=list(results.keys()),
        y=[metrics['total_weight_per_shift'] for metrics in results.values()],
        title="Общий принятый вес за смену по зонам",
        labels={'x': 'Температурная зона', 'y': 'Вес (кг)'},
        color=list(results.keys())
    )
    fig_weight.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_weight, use_container_width=True)


def main():
    """
    Основная функция веб-приложения
    """
    # Заголовок приложения
    st.title("🏭 Планирование приёмки по температурным зонам")
    st.markdown("""
    Система для расчета производительности склада по различным температурным зонам.
    Используется машинное обучение для предсказания времени приёмки и расчета пропускной способности.
    """)

    # Попытка загрузки моделей
    trainer, data_processor = load_trained_models()

    # Определение статуса моделей
    if trainer is not None and data_processor is not None:
        model_status = "✅ Модели загружены успешно"
        use_real_models = True
        available_zones = trainer.get_available_temp_zones()
        st.success(f"{model_status}. Доступные зоны: {', '.join(available_zones)}")
    else:
        model_status = "⚠️ Модели не найдены. Используется демонстрационный режим."
        use_real_models = False
        st.warning(model_status)
        st.info("Для использования обученных моделей запустите сначала: `python model_training.py`")

    # Создание элементов ввода
    zone_inputs, temp_zones = create_temp_zone_inputs()
    gates, unload_time, shift_hours = create_additional_inputs()

    # Отображение текущих значений в боковой панели
    # with st.sidebar:
    #     st.header("📋 Текущие параметры")
    #
    #     st.subheader("Количество ворот")
    #     for zone, count in gates.items():
    #         st.write(f"**{zone}:** {count}")
    #
    #     st.subheader("Временные параметры")
    #     st.write(f"**Время разгрузки ТС:** {unload_time} мин")
    #     st.write(f"**Часов в смене:** {shift_hours} ч")
    #
    #     if st.button("🔄 Обновить приложение"):
    #         st.cache_resource.clear()
    #         st.rerun()

    # Кнопка для расчета
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        calculate_button = st.button(
            "🔮 Рассчитать производительность склада",
            type="primary",
            use_container_width=True
        )

    if calculate_button:
        with st.spinner('Выполняется расчёт...'):
            # Расчет среднего времени приёмки для каждой зоны
            avg_times = {}

            for zone in temp_zones:
                # Подготовка данных для предсказания
                input_df = pd.DataFrame([zone_inputs[zone]])

                # Выполнение предсказания
                if use_real_models and zone in trainer.get_available_temp_zones():
                    prediction = trainer.predict_for_temp_zone(input_df, zone)
                else:
                    prediction = create_demo_prediction(zone_inputs[zone], zone)

                avg_times[zone] = prediction if prediction is not None else 30.0

            # Расчет метрик производительности
            results = calculate_capacity_metrics(avg_times, gates, unload_time, shift_hours, zone_inputs)

        st.success("Расчёт выполнен успешно!")

        # Отображение результатов
        create_results_visualization(results)

        # Дополнительная информация
        with st.expander("ℹ️ Методология расчетов"):
            st.markdown("""
            **Формулы расчета:**

            1. **Количество приёмок за смену:**
               ```
               Количество ворот × Часы в смене × 60 / (Время разгрузки ТС + Среднее время приёмки)
               ```

            2. **Общий принятый вес за смену:**
               ```
               Вес приёмки × Количество ворот × Часы в смене × 60 / (Время разгрузки ТС + Среднее время приёмки)
               ```

            **Особенности расчета:**
            - Время приёмки рассчитывается отдельной моделью для каждой температурной зоны
            - Учитывается время разгрузки транспортного средства
            - Расчет ведется на основе доступного количества ворот в каждой зоне
            """)

            if use_real_models:
                st.markdown("""
                **О моделях машинного обучения:**
                - Для каждой температурной зоны обучена отдельная модель
                - Модели учитывают количество SKU, принятое количество, вес и объём
                - Используется автоматический выбор лучшего алгоритма (AutoML)
                """)
            else:
                st.markdown("""
                **Демонстрационный режим:**
                - Используются упрощенные формулы для расчета времени приёмки
                - Для точных расчетов необходимо обучить модели на реальных данных
                """)


if __name__ == "__main__":
    main()
