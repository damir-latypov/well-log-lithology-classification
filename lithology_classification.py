
"""Классификация литологии по данным ГИС

Модель машинного обучения для автоматического определения типа горной породы
по данным геофизических исследований скважин.
"""

# =============================================================================
# ИМПОРТ БИБЛИОТЕК
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io
import warnings
import shap
import os
import sys

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                           classification_report, confusion_matrix)

# Настройка визуализации
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# =============================================================================
# СОЗДАНИЕ ПАПОК ДЛЯ РЕЗУЛЬТАТОВ
# =============================================================================
print("📁 Создание структуры папок...")

# Создаем необходимые папки
folders = ['images', 'results', 'data']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"✅ Создана папка: {folder}/")
    else:
        print(f"📁 Папка уже существует: {folder}/")

# =============================================================================
# ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# =============================================================================
print("\n🔄 Загрузка данных...")

try:
    # Скачиваем и распаковываем напрямую в память
    zip_url = "https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/raw/master/lithology_competition/data/train.zip"
    r = requests.get(zip_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Читаем CSV из архива
    df = pd.read_csv(z.open('train.csv'), sep=';')
    print("✅ Данные успешно загружены")
except Exception as e:
    print(f"❌ Ошибка загрузки данных: {e}")
    sys.exit(1)

# Перевод названий колонок на русский
russian_columns = {
    'WELL': 'СКВАЖИНА',
    'DEPTH_MD': 'ГЛУБИНА_ИЗМЕРЕННАЯ',
    'X_LOC': 'КООРДИНАТА_X',
    'Y_LOC': 'КООРДИНАТА_Y',
    'Z_LOC': 'КООРДИНАТА_Z',
    'GROUP': 'ГРУППА',
    'FORMATION': 'ФОРМАЦИЯ',
    'CALI': 'КАЛИБР',
    'RSHA': 'RES_ЗОНА_ВТОРЖЕНИЯ',
    'RMED': 'RES_СРЕДНЯЯ',
    'RDEP': 'RES_ГЛУБОКАЯ',
    'RHOB': 'ПЛОТНОСТЬ',
    'GR': 'ГК',
    'SGR': 'ГК_СПЕКТРАЛЬНЫЙ',
    'NPHI': 'ННК',
    'PEF': 'PEF',
    'DTC': 'DTC',
    'SP': 'СП',
    'BS': 'РАЗМЕР_ДОЛОТА',
    'ROP': 'СКОРОСТЬ_ПРОХОДКИ',
    'DTS': 'DTS',
    'DCAL': 'КАЛИБР_РАСХОЖДЕНИЕ',
    'DRHO': 'ПЛОТНОСТЬ_ПОПРАВКА',
    'MUDWEIGHT': 'ПЛОТНОСТЬ_РАСТВОРА',
    'RMIC': 'RES_МИКРО',
    'ROPA': 'RES_????',
    'RXO': 'RES_ЗОНА_ПРОНИКНОВЕНИЯ',
    'FORCE_2020_LITHOFACIES_LITHOLOGY': 'ЛИТОЛОГИЯ',
    'FORCE_2020_LITHOFACIES_CONFIDENCE': 'ДОСТОВЕРНОСТЬ_ЛИТОЛОГИИ'
}

df = df.rename(columns=russian_columns)
target = 'ЛИТОЛОГИЯ'
df = df.drop('ДОСТОВЕРНОСТЬ_ЛИТОЛОГИИ', axis=1)

# =============================================================================
# EDA - АНАЛИЗ ДАННЫХ
# =============================================================================
print("\n📊 БАЗОВЫЙ АНАЛИЗ ДАННЫХ")
print(f"Размер датасета: {df.shape}")
print(f"Количество скважин: {df['СКВАЖИНА'].nunique()}")
print(f"Диапазон глубин: {df['ГЛУБИНА_ИЗМЕРЕННАЯ'].min():.1f} - {df['ГЛУБИНА_ИЗМЕРЕННАЯ'].max():.1f} м")

# Анализ пропущенных значений
print("\n🔍 АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
miss_data = df.isnull().sum()
miss_per = (miss_data/len(df))*100
miss_info = pd.DataFrame({
    'Количество': miss_data,
    'Процент': miss_per
}).sort_values('Количество', ascending=False)
miss_info = miss_info[miss_info['Количество'] > 0]
print(miss_info)

# Сохраняем анализ пропущенных значений
miss_info.to_csv('results/missing_values_analysis.csv', encoding='utf-8-sig')
print("✅ Анализ пропущенных значений сохранен в results/missing_values_analysis.csv")

# Удаление признаков с >80% пропусков
cols_todrop = miss_info[miss_info['Процент'] > 80].index.tolist()
if cols_todrop:
    print(f"🗑️ Удаляем признаки с >80% пропусков: {cols_todrop}")
    df = df.drop(cols_todrop, axis=1)
else:
    print("✅ Нет признаков с >80% пропусков")

# Заполнение пропусков
features_tofill = ['ННК', 'ПЛОТНОСТЬ', 'КАЛИБР', 'RES_ГЛУБОКАЯ', 'DTC']
print(f"\n🔄 Заполнение пропусков для признаков: {features_tofill}")

for f in features_tofill:
    if f in df.columns:
        df[f] = df.groupby('СКВАЖИНА')[f].transform(lambda x: x.fillna(x.median()))
        print(f"✅ Заполнены пропуски для {f}")

if 'КАЛИБР' in df.columns:
    df['КАЛИБР'] = df['КАЛИБР'].fillna(df['КАЛИБР'].median())
    print("✅ Заполнены оставшиеся пропуски в КАЛИБР")

# =============================================================================
# ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ
# =============================================================================
print("\n📈 ВИЗУАЛИЗАЦИЯ ДАННЫХ...")

# Создаем график распределения основных признаков
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
main_features = ['ГК', 'ПЛОТНОСТЬ', 'RES_ГЛУБОКАЯ', 'ННК', 'DTC', 'КАЛИБР']

for i, feature in enumerate(main_features):
    if feature in df.columns:
        ax = axes[i//3, i%3]
        df[feature].hist(bins=50, ax=ax, alpha=0.7)
        ax.set_title(f'Распределение {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Частота')

plt.tight_layout()
plt.savefig('images/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Графики распределений сохранены в images/feature_distributions.png")

# =============================================================================
# ПОДГОТОВКА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# =============================================================================
lithology_dict = {
    65000: 'Песчаник',
    30000: 'Песчаник-Глина',
    65030: 'Глинистый песчаник',
    70000: 'Известняк',
    80000: 'Мергель',
    88000: 'Известковистая глина',
    90000: 'Доломит',
    74000: 'Мел',
    86000: 'Глина',
    93000: 'Туф',
    70032: 'Песчанистый известняк',
    99000: 'Ангидрит'
}

df['ЛИТОЛОГИЯ_ТЕКСТ'] = df['ЛИТОЛОГИЯ'].map(lithology_dict)

print("\n🎯 АНАЛИЗ ЛИТОЛОГИИ")
print(f"Количество классов: {df['ЛИТОЛОГИЯ_ТЕКСТ'].nunique()}")
print("Распределение классов:")
lithology_counts = df['ЛИТОЛОГИЯ_ТЕКСТ'].value_counts()
print(lithology_counts)

# Визуализация распределения литологии
plt.figure(figsize=(12, 6))
lithology_counts.plot(kind='bar', color='skyblue')
plt.title('Распределение типов литологии')
plt.xlabel('Тип литологии')
plt.ylabel('Количество образцов')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/lithology_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Распределение литологии сохранено в images/lithology_distribution.png")

# Сохраняем статистику по литологии
lithology_counts.to_csv('results/lithology_statistics.csv', encoding='utf-8-sig')

# Объединение редких классов
lithology_merge = {
    65030: 30000,  # Глинистый песчаник → Песчаник-Глина
    86000: 88000,  # Глина → Известковистая глина
    74000: 70000,  # Мел → Известняк
    70032: 70000,  # Песчанистый известняк → Известняк
    93000: 70000,  # Туф → Известняк
}

# Работаем с подвыборкой для скорости
df_sample = df.sample(frac=0.1, random_state=42)
df_sample['ЛИТОЛОГИЯ'] = df_sample['ЛИТОЛОГИЯ'].replace(lithology_merge)

print(f"\n📊 Размер подвыборки для обучения: {df_sample.shape}")

# =============================================================================
# FEATURE ENGINEERING - СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ
# =============================================================================
print("\n🔧 СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ...")

# Геофизические отношения
df_sample['ГК/ПЛОТНОСТЬ'] = df_sample['ГК'] / df_sample['ПЛОТНОСТЬ']
df_sample['ННК*ПЛОТНОСТЬ'] = df_sample['ННК'] * df_sample['ПЛОТНОСТЬ']
df_sample['RES_ГЛУБОКАЯ/RES_СРЕДНЯЯ'] = df_sample['RES_ГЛУБОКАЯ'] / df_sample['RES_СРЕДНЯЯ']

# Скользящие статистики
df_sample['ГК_МА'] = df_sample.groupby('СКВАЖИНА')['ГК'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)
df_sample['ПЛОТНОСТЬ_STD'] = df_sample.groupby('СКВАЖИНА')['ПЛОТНОСТЬ'].transform(
    lambda x: x.rolling(window=5, min_periods=1).std()
)

# Разности
df_sample['ГК_DIFF'] = df_sample.groupby('СКВАЖИНА')['ГК'].diff()
df_sample['ПЛОТНОСТЬ_DIFF'] = df_sample.groupby('СКВАЖИНА')['ПЛОТНОСТЬ'].diff()

# Заполняем пропуски в новых признаках
new_features = ['ГК/ПЛОТНОСТЬ', 'ННК*ПЛОТНОСТЬ', 'RES_ГЛУБОКАЯ/RES_СРЕДНЯЯ',
                'ГК_DIFF', 'ПЛОТНОСТЬ_DIFF', 'ГК_МА', 'ПЛОТНОСТЬ_STD']

for feature in new_features:
    df_sample[feature] = df_sample[feature].fillna(df_sample[feature].median())

print(f"✅ Создано {len(new_features)} новых признаков")

# =============================================================================
# ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
# =============================================================================
# Удаляем неинформативные и целевые признаки
columns_to_drop = ['ЛИТОЛОГИЯ', 'ЛИТОЛОГИЯ_ТЕКСТ', 'СКВАЖИНА', 'ГРУППА',
                   "ФОРМАЦИЯ", 'КАЛИБР_РАСХОЖДЕНИЕ', 'ПЛОТНОСТЬ_РАСТВОРА',
                   'СКОРОСТЬ_ПРОХОДКИ', 'RES_ЗОНА_ПРОНИКНОВЕНИЯ']

# Проверяем, какие колонки действительно существуют
existing_columns_to_drop = [col for col in columns_to_drop if col in df_sample.columns]
X = df_sample.drop(existing_columns_to_drop, axis=1)
y = df_sample['ЛИТОЛОГИЯ']

print(f"✅ Признаки для обучения: {X.shape[1]}")
print(f"✅ Целевая переменная: {y.nunique()} классов")

# Кодируем целевую переменную
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\n📝 СООТВЕТСТВИЕ КЛАССОВ:")
for i, class_name in enumerate(le.classes_):
    print(f"{i:2d} -> {class_name:6} ({lithology_dict[class_name]})")

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📊 РАЗБИЕНИЕ ДАННЫХ:")
print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# =============================================================================
# МОДЕЛЬ XGBOOST С НОВЫМИ ПРИЗНАКАМИ
# =============================================================================
print("\n🚀 ОБУЧЕНИЕ XGBOOST С FEATURE ENGINEERING...")

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# Предсказания и метрики
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"✅ Accuracy: {accuracy:.3f}")
print(f"✅ F1-score (macro): {f1:.3f}")

print("\n📈 ДЕТАЛЬНЫЙ ОТЧЕТ:")
print(classification_report(y_test, y_pred,
                          target_names=[lithology_dict[le.classes_[i]] for i in range(len(le.classes_))]))

# =============================================================================
# АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
# =============================================================================
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 ТОП-10 САМЫХ ВАЖНЫХ ПРИЗНАКОВ:")
print(importances.head(10))

# Сохраняем важность признаков
importances.to_csv('results/feature_importance.csv', encoding='utf-8-sig', index=False)
print("✅ Важность признаков сохранена в results/feature_importance.csv")

# Визуализация важности признаков
plt.figure(figsize=(10, 8))
top_features = importances.head(15)
sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
plt.title('Топ-15 самых важных признаков')
plt.xlabel('Важность')
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ График важности признаков сохранен в images/feature_importance.png")

# Анализ новых фичей
print(f"\n🎯 РЕЙТИНГ НОВЫХ ПРИЗНАКОВ:")
new_features_rank = []
for feature in new_features:
    if feature in importances['feature'].values:
        rank = importances[importances['feature'] == feature].index[0] + 1
        imp = importances[importances['feature'] == feature]['importance'].values[0]
        new_features_rank.append((rank, feature, imp))
        print(f"{rank:2d}. {feature:25} (важность: {imp:.4f})")

# =============================================================================
# SHAP АНАЛИЗ
# =============================================================================
print("\n🧠 SHAP АНАЛИЗ...")

try:
    # Берем подвыборку для скорости
    sample_idx = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Визуализация важности признаков
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig('images/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ SHAP анализ сохранен в images/shap_feature_importance.png")

    # Детальный SHAP plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('images/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Детальный SHAP plot сохранен в images/shap_summary_plot.png")

except Exception as e:
    print(f"⚠️ SHAP анализ не удался: {e}")
    print("Продолжаем без SHAP...")

# =============================================================================
# ОПТИМИЗАЦИЯ МОДЕЛИ
# =============================================================================
print("\n⚡ ОПТИМИЗАЦИЯ МОДЕЛИ...")

# Удаляем слабые признаки (важность < 0.01)
features_to_drop = []
for feature in new_features:
    if feature in importances['feature'].values:
        imp = importances[importances['feature'] == feature]['importance'].values[0]
        if imp < 0.01:
            features_to_drop.append(feature)

if features_to_drop:
    print(f"🗑️ Удаляем слабые признаки: {features_to_drop}")
    X_clean = X.drop(features_to_drop, axis=1)
else:
    print("✅ Нет слабых признаков для удаления")
    X_clean = X.copy()

# Переразделяем данные с очищенными признаками
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Финальная модель
xgb_final = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_final.fit(X_train_clean, y_train_clean)

# Финальные метрики
y_pred_final = xgb_final.predict(X_test_clean)
final_accuracy = accuracy_score(y_test_clean, y_pred_final)
final_f1 = f1_score(y_test_clean, y_pred_final, average='macro')

print(f"🎯 ФИНАЛЬНЫЙ ACCURACY: {final_accuracy:.3f}")
print(f"🎯 ФИНАЛЬНЫЙ F1-SCORE: {final_f1:.3f}")

# Матрица ошибок
cm = confusion_matrix(y_test_clean, y_pred_final)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[lithology_dict[cls] for cls in le.classes_],
            yticklabels=[lithology_dict[cls] for cls in le.classes_])
plt.title('Матрица ошибок - Финальная модель')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Матрица ошибок сохранена в images/confusion_matrix.png")

# =============================================================================
# ФИНАЛЬНЫЙ ОТЧЕТ И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================
print("\n" + "="*60)
print("🎉 ФИНАЛЬНЫЙ ОТЧЕТ ПРОЕКТА")
print("="*60)

final_importances = pd.DataFrame({
    'feature': X_clean.columns,
    'importance': xgb_final.feature_importances_
}).sort_values('importance', ascending=False)

print(f"Финальный accuracy (оптимизация): {final_accuracy:.3f}")
print(f"Количество признаков: {len(X_clean.columns)}")

print("\n🔝 ТОП-5 САМЫХ ВАЖНЫХ ПРИЗНАКОВ:")
print(final_importances.head(5))

# Анализ новых признаков в топе
new_features_in_top = []
for feature in ['ГК/ПЛОТНОСТЬ', 'ННК*ПЛОТНОСТЬ', 'ГК_МА', 'ПЛОТНОСТЬ_STD']:
    if feature in final_importances['feature'].values:
        rank = final_importances[final_importances['feature'] == feature].index[0] + 1
        imp = final_importances[final_importances['feature'] == feature]['importance'].values[0]
        new_features_in_top.append((feature, rank, imp))

print(f"\n🎯 Новые признаки в топе: {len(new_features_in_top)}")
for feature, rank, imp in new_features_in_top:
    print(f"   {rank}. {feature} (важность: {imp:.4f})")

# Сохраняем финальные результаты
results_summary = {
    'final_accuracy': final_accuracy,
    'final_f1_score': final_f1,
    'num_features': len(X_clean.columns),
    'top_features': final_importances.head(10)['feature'].tolist()
}

import json
with open('results/final_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

print("\n💾 ВСЕ РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
print("   📊 images/ - графики и визуализации")
print("   📈 results/ - метрики и анализ")
print("   ✅ requirements.txt - зависимости проекта")

print("\n✅ ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")
