import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
import pickle
import os
import pymorphy2
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import time

# Импорты TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(f"TensorFlow version: {tf.__version__}")


# Инициализация лемматизатора
@st.cache_resource
def load_morph():
    """Загрузка лемматизатора pymorphy2"""
    try:
        morph = pymorphy2.MorphAnalyzer()
        return morph
    except Exception as e:
        st.warning(f"Не удалось загрузить pymorphy2: {e}")
        return None


st.set_page_config(
    page_title="Классификация токсичных комментариев",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Функции для работы с MySQL
@st.cache_resource
def init_connection():
    """Инициализация подключения к MySQL"""
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Root",
            database="comments"
        )
        return connection
    except Error as e:
        st.error(f"Ошибка подключения к MySQL: {e}")
        return None


def save_prediction_to_db(connection, comment_text, class_name, probability):
    """Сохранение результата предсказания в БД"""
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO comments (comment, name_class, procent)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (comment_text, class_name, float(probability)))
        connection.commit()
        cursor.close()
        return True
    except Error as e:
        st.error(f"Ошибка при сохранении в БД: {e}")
        return False


def get_last_predictions(connection, limit=3):
    """Получение последних предсказаний из БД"""
    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT id_comment, comment, name_class, procent 
        FROM comments 
        ORDER BY id_comment DESC 
        LIMIT %s
        """
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        cursor.close()
        return results
    except Error as e:
        st.error(f"Ошибка при получении данных из БД: {e}")
        return []


def get_prediction_stats(connection):
    """Получение статистики предсказаний из БД"""
    try:
        cursor = connection.cursor(dictionary=True)

        # Общее количество предсказаний
        cursor.execute("SELECT COUNT(*) as total FROM comments")
        total = cursor.fetchone()['total']

        # Статистика по классам
        cursor.execute("""
            SELECT name_class, COUNT(*) as count, AVG(procent) as avg_procent
            FROM comments 
            GROUP BY name_class
        """)
        class_stats = cursor.fetchall()

        cursor.close()
        return total, class_stats
    except Error as e:
        st.error(f"Ошибка при получении статистики: {e}")
        return 0, []


# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #2c3e50 !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }
    .warning-card {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .class-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .badge-normal {
        background: #00c853;
        color: white;
    }
    .badge-insult {
        background: #ff5252;
        color: white;
    }
    .badge-threat {
        background: #ff9800;
        color: white;
    }
    .badge-obscenity {
        background: #9c27b0;
        color: white;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .history-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .history-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    .db-stats-card {
        background: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Функции предобработки
@st.cache_resource
def load_ml_components():
    """Загрузка модели, токенизатора и параметров"""
    try:
        # Проверяем наличие файлов
        required_files = ['toxicity_model.h5', 'tokenizer.pickle', 'model_params.pickle', 'target_columns.pickle']
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            st.error(f"❌ Отсутствуют файлы: {', '.join(missing_files)}")
            return None, None, None, None

        # Загружаем модель
        with st.spinner("Загрузка модели..."):
            model = load_model('toxicity_model.h5')

        # Загружаем токенизатор
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Загружаем параметры
        with open('model_params.pickle', 'rb') as handle:
            params = pickle.load(handle)

        # Загружаем названия классов
        with open('target_columns.pickle', 'rb') as handle:
            target_columns = pickle.load(handle)

        return model, tokenizer, params, target_columns
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None, None, None


def lemmatize_text(text, morph):
    """Лемматизация текста с помощью pymorphy2"""
    if morph is None:
        return text

    words = text.split()
    lemmatized_words = []

    for word in words:
        # Пропускаем специальные токены (URL и т.д.)
        if word in ['url', 'URL']:
            lemmatized_words.append(word)
            continue

        try:
            # Лемматизация слова
            parsed = morph.parse(word)[0]
            lemmatized_words.append(parsed.normal_form)
        except:
            # В случае ошибки оставляем исходное слово
            lemmatized_words.append(word)

    return ' '.join(lemmatized_words)


def preprocess_text(text, morph=None, use_lemmatization=True):
    """Предобработка текста с опциональной лемматизацией"""
    if not isinstance(text, str):
        text = str(text)

    # Приведение к нижнему регистру
    text = text.lower()

    # Замена "ё" на "е"
    text = text.replace("ё", "е")

    # Замена URL на специальный токен
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' URL ', text)

    # Удаление всех символов кроме букв и пробелов
    text = re.sub(r'[^a-zA-Zа-яА-Я\s]+', ' ', text)

    # Удаление лишних пробелов
    text = re.sub(r' +', ' ', text)

    # Лемматизация (если включена)
    if use_lemmatization and morph is not None:
        text = lemmatize_text(text, morph)

    return text.strip()


def predict_comment(model, tokenizer, text, morph, maxlen=300, use_lemmatization=True):
    """Предсказание для одного комментария с опциональной лемматизацией"""
    try:
        # Предобработка с лемматизацией
        processed_text = preprocess_text(text, morph, use_lemmatization)

        # Токенизация
        sequence_text = tokenizer.texts_to_sequences([processed_text])

        # Паддинг
        padded = pad_sequences(sequence_text, maxlen=maxlen)

        # Предсказание
        prediction = model.predict(padded, verbose=0)

        return prediction[0], processed_text
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
        return None, None


# Заголовок приложения
st.markdown('<div class="main-header">💬 Классификация токсичных комментариев</div>',
            unsafe_allow_html=True)

# Загрузка модели и лемматизатора
model, tokenizer, params, target_columns = load_ml_components()
morph = load_morph()

# Подключение к MySQL
db_connection = init_connection()

# Боковая панель
with st.sidebar:
    st.markdown("## ℹ️ О приложении")
    st.markdown("""
    Это приложение использует **BiGRU нейронную сеть** для классификации комментариев на 4 категории:
    """)

    # Цветные метки для категорий
    st.markdown("""
    <span class="class-badge badge-normal">normal</span> - нормальный
    <span class="class-badge badge-insult">insult</span> - оскорбление
    <span class="class-badge badge-threat">threat</span> - угроза
    <span class="class-badge badge-obscenity">obscenity</span> - нецензурная лексика
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Статус подключения к БД
    st.markdown("### 💾 База данных")
    if db_connection and db_connection.is_connected():
        st.success("✅ Подключено к MySQL")

        # Получаем статистику из БД
        total_predictions, class_stats = get_prediction_stats(db_connection)

        # Отображаем статистику
        st.markdown(f"**Всего предсказаний:** {total_predictions}")

        if class_stats:
            st.markdown("**Статистика по классам:**")
            for stat in class_stats:
                color = {
                    'normal': '#00c853',
                    'insult': '#ff5252',
                    'threat': '#ff9800',
                    'obscenity': '#9c27b0'
                }.get(stat['name_class'], '#667eea')

                st.markdown(f"""
                <div style="background: {color}; color: white; padding: 0.5rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>{stat['name_class']}:</strong> {stat['count']} предсказаний 
                    (ср. уверенность: {stat['avg_procent']:.2f}%)
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("❌ Не подключено к MySQL")
        if st.button("🔄 Попробовать переподключиться"):
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")

    # Настройки предобработки
    st.markdown("### ⚙️ Настройки обработки")

    use_lemmatization = st.checkbox(
        "Использовать лемматизацию",
        value=True,
        help="Приведение слов к нормальной форме (работает медленнее, но точнее)"
    )

    if morph is None:
        st.warning("⚠️ Лемматизатор не загружен. Опция отключена.")
        use_lemmatization = False

    st.markdown("---")

    if model is not None:
        st.success("✅ Модель загружена")
        if params:
            st.info("📊 **Параметры модели:**")
            st.json(params)

    # Метрики модели
    st.markdown("### 📊 Метрики модели")
    st.markdown("""
    | Метрика | Значение |
    |---------|----------|
    | Точность | 96.9% |
    | Полнота | 97.5% |
    | AUC | 0.998 |
    """)

    # Примеры комментариев
    st.markdown("### 📝 Примеры комментариев")
    examples = [
        "Это отличный пост, спасибо за информацию!",
        "Ты просто дурак, иди учи матчасть",
        "Я тебя найду и накажу",
        "За такие слова можно и по лицу получить",
        "Отличная работа, продолжайте в том же духе",
        "ебанные нубы заходите сервер"
    ]

    for i, example in enumerate(examples):
        if st.button(f"📋 Пример {i + 1}", key=f"example_{i}", use_container_width=True):
            st.session_state.input_text = example
            st.rerun()

# Основная область
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### ✍️ Введите комментарий")

    # Текстовая область для ввода
    input_text = st.text_area(
        "Введите текст комментария для анализа:",
        value=st.session_state.get('input_text', ''),
        height=150,
        placeholder="Напишите комментарий здесь...",
        key="comment_input"
    )

    # Обработанный текст
    if input_text:
        with st.expander("🔍 Этапы обработки"):
            # Показываем разные этапы обработки
            col_stage1, col_stage2 = st.columns(2)

            with col_stage1:
                st.markdown("**📝 Исходный текст:**")
                st.write(input_text)

                st.markdown("**🔽 После базовой обработки:**")
                basic_processed = preprocess_text(input_text, None, False)
                st.write(basic_processed)

            with col_stage2:
                if use_lemmatization and morph is not None:
                    st.markdown("**🔄 После лемматизации:**")
                    lemmatized = preprocess_text(input_text, morph, True)
                    st.write(lemmatized)

                    # Показываем примеры лемматизации
                    st.markdown("**📚 Примеры лемматизации:**")
                    examples_text = "красивый красивая красивые красивее"
                    st.write(f"Исходные формы: {examples_text}")
                    st.write(f"Леммы: {lemmatize_text(examples_text, morph)}")

    # Кнопки управления
    col_bt1, col_bt2 = st.columns(2)

    with col_bt1:
        predict_btn = st.button("🚀 Предсказать", use_container_width=True, type="primary")

    with col_bt2:
        if st.button("🧹 Очистить", use_container_width=True):
            st.session_state.input_text = ""
            if 'prediction_result' in st.session_state:
                del st.session_state.prediction_result
            st.rerun()

    # Информация о предобработке
    with st.expander("ℹ️ Как работает предобработка"):
        st.markdown("""
        **Этапы предобработки:**
        1. Приведение к нижнему регистру
        2. Замена 'ё' на 'е'
        3. Замена URL на 'URL'
        4. Удаление всех символов кроме букв
        5. Удаление лишних пробелов
        6. **Лемматизация** (опционально) - приведение слов к начальной форме
           - Например: "красивые", "красивого" → "красивый"
           - "бегал", "бегает" → "бегать"
        """)

with col2:
    st.markdown("### 📊 Результаты классификации")

    if predict_btn and input_text and model is not None:
        with st.spinner("Анализ комментария..."):
            # Получаем предсказание с учетом лемматизации
            prediction, processed_text = predict_comment(
                model, tokenizer, input_text, morph,
                params.get('maxlen', 300),
                use_lemmatization
            )

            if prediction is not None:
                # Определяем класс и уверенность
                predicted_idx = np.argmax(prediction)
                predicted_class = target_columns[predicted_idx]
                confidence = prediction[predicted_idx] * 100

                # Сохраняем результат в БД
                if db_connection and db_connection.is_connected():
                    save_success = save_prediction_to_db(
                        db_connection,
                        input_text,
                        predicted_class,
                        confidence
                    )
                    if save_success:
                        st.success("✅ Результат сохранен в базу данных")
                    else:
                        st.warning("⚠️ Не удалось сохранить результат в БД")

                # Сохраняем результат в session_state
                st.session_state.prediction_result = {
                    'text': input_text,
                    'processed': processed_text,
                    'predictions': prediction,
                    'classes': target_columns,
                    'used_lemmatization': use_lemmatization
                }

                # Очищаем кэш статистики БД для обновления
                st.cache_resource.clear()

    # Отображение результатов
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        predictions = result['predictions']
        classes = result['classes']

        # Основное предсказание
        predicted_idx = np.argmax(predictions)
        predicted_class = classes[predicted_idx]
        confidence = predictions[predicted_idx] * 100

        # Цвет карточки в зависимости от класса
        if predicted_class == 'normal':
            gradient = "linear-gradient(135deg, #00c853 0%, #64dd17 100%)"
        elif predicted_class == 'insult':
            gradient = "linear-gradient(135deg, #ff5252 0%, #ff1744 100%)"
        elif predicted_class == 'threat':
            gradient = "linear-gradient(135deg, #ff9800 0%, #ff6517 100%)"
        else:  # obscenity
            gradient = "linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%)"

        # Карточка результата
        st.markdown(f"""
        <div class="prediction-card" style="background: {gradient};">
            <h2>🎯 Результат анализа</h2>
            <h3 style="margin: 1rem 0;">{predicted_class.upper()}</h3>
            <h1 style="font-size: 3rem; margin: 0.5rem 0;">{confidence:.1f}%</h1>
            <p>уверенность</p>
        </div>
        """, unsafe_allow_html=True)

        # Предупреждение для токсичных комментариев
        if predicted_class != 'normal':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ ВНИМАНИЕ!</strong> Комментарий содержит {predicted_class} и может быть токсичным.
            </div>
            """, unsafe_allow_html=True)

        # Информация об обработке
        if result.get('used_lemmatization', False):
            st.info("✅ Лемматизация была применена к тексту")

        # Таблица всех предсказаний
        st.markdown("### 📈 Детальные предсказания")

        # Создаем DataFrame
        df_predictions = pd.DataFrame({
            'Категория': classes,
            'Вероятность (%)': [p * 100 for p in predictions],
            'Статус': ['✅' if i == predicted_idx else '○' for i in range(len(classes))]
        })

        # Форматируем проценты
        df_predictions['Вероятность (%)'] = df_predictions['Вероятность (%)'].round(2)

        # Сортируем по вероятности
        df_predictions = df_predictions.sort_values('Вероятность (%)', ascending=False)

        # Отображаем таблицу
        st.dataframe(
            df_predictions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Категория": "Категория",
                "Вероятность (%)": st.column_config.NumberColumn(
                    "Вероятность (%)",
                    format="%.2f%%"
                ),
                "Статус": "Статус"
            }
        )

        # Визуализация
        st.markdown("### 📊 Визуализация предсказаний")

        # Создаем горизонтальную бар-чарт
        fig = px.bar(
            df_predictions,
            x='Вероятность (%)',
            y='Категория',
            orientation='h',
            color='Вероятность (%)',
            color_continuous_scale='RdYlGn',
            title='Вероятность для каждой категории'
        )

        # Добавляем вертикальную линию для 50%
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            xaxis_title="Вероятность (%)",
            yaxis_title="Категория",
            height=300,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Текст комментария для контекста
        with st.expander("📝 Текст после обработки"):
            st.write(result['processed'])

    elif model is None:
        st.error("❌ Модель не загружена. Проверьте файлы в директории:")
        st.code("""
        Необходимые файлы:
        - toxicity_model.h5
        - tokenizer.pickle
        - model_params.pickle
        - target_columns.pickle
        """)

        # Показываем текущую директорию
        st.info(f"📁 Текущая директория: {os.getcwd()}")
        st.info(f"📄 Файлы в директории: {os.listdir('.')}")
    elif not input_text:
        st.info("👆 Введите комментарий для анализа")

# История предсказаний
st.markdown("---")
st.markdown("### 📜 Последние предсказания")

if db_connection and db_connection.is_connected():
    last_predictions = get_last_predictions(db_connection, 3)

    if last_predictions:
        cols = st.columns(3)
        for i, pred in enumerate(last_predictions):
            with cols[i]:
                # Цвет в зависимости от класса
                border_color = {
                    'normal': '#00c853',
                    'insult': '#ff5252',
                    'threat': '#ff9800',
                    'obscenity': '#9c27b0'
                }.get(pred['name_class'], '#667eea')

                st.markdown(f"""
                <div class="history-card" style="border-left-color: {border_color};">
                    <strong>ID: {pred['id_comment']}</strong><br>
                    <span style="color: {border_color}; font-weight: bold;">{pred['name_class'].upper()}</span><br>
                    <small>Уверенность: {pred['procent']:.2f}%</small><br>
                    <small style="color: #666;">"{pred['comment'][:50]}..."</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📭 В базе данных пока нет предсказаний")
else:
    st.warning("⚠️ База данных недоступна. История предсказаний не отображается")

# Дополнительная информация внизу
st.markdown("---")

# Статистика модели
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.markdown("""
    <div class="stats-card">
        <div class="metric-value">96.9%</div>
        <div class="metric-label">Точность</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat2:
    st.markdown("""
    <div class="stats-card">
        <div class="metric-value">97.5%</div>
        <div class="metric-label">Полнота</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat3:
    st.markdown("""
    <div class="stats-card">
        <div class="metric-value">0.998</div>
        <div class="metric-label">AUC</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat4:
    st.markdown("""
    <div class="stats-card">
        <div class="metric-value">300</div>
        <div class="metric-label">Макс. длина</div>
    </div>
    """, unsafe_allow_html=True)

# Футер
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #666; margin-top: 2rem;">
    © 2025, Классификация токсичных комментариев • BiGRU + FastText embeddings + Лемматизация + MySQL
</p>
""", unsafe_allow_html=True)