from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data.csv"
FUZZY_PATH = BASE_DIR / "fuzzy_results.csv"
GRAPH_DIR = BASE_DIR / "graphs"

FEATURES = [
    "revenue",
    "costs",
    "employees",
    "productivity",
    "overdue_orders",
    "defects",
    "plan_percent",
    "profitability",
    "revenue_per_employee",
]

TARGET = "state"

CLASS_NAMES = [
    "Проблемное",
    "Требует внимания",
    "Устойчивое",
]

STATE_NAMES = {
    0: "Проблемное",
    1: "Требует внимания",
    2: "Устойчивое",
}


def train_ml_model(df):
    """Обучает нейронную сеть и возвращает тестовые результаты."""

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        ),
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred


def main():
    if not DATA_PATH.exists():
        print("Не найден файл data.csv.")
        return

    if not FUZZY_PATH.exists():
        print("Не найден файл fuzzy_results.csv.")
        print("Сначала запусти: python fuzzy_model.py")
        return

    df = pd.read_csv(DATA_PATH)
    fuzzy_df = pd.read_csv(FUZZY_PATH)

    # Проверяем, что нечёткие результаты относятся к тем же записям.
    if len(df) != len(fuzzy_df):
        print("Количество записей в файлах не совпадает.")
        return

    model, X_test, y_test, y_pred = train_ml_model(df)

    # Индексы тестовых записей сохраняются при разделении выборки.
    test_indices = X_test.index

    # Берём нечёткие прогнозы для тех же тестовых записей.
    fuzzy_test = fuzzy_df.loc[test_indices, "fuzzy_state"]

    fuzzy_mapping = {
        "Проблемное": 0,
        "Требует внимания": 1,
        "Устойчивое": 2,
    }

    y_fuzzy = fuzzy_test.map(fuzzy_mapping)

    # Для сравнения учитываем записи с числовым прогнозом нечёткой модели.
    valid = y_fuzzy.notna()

    y_true_valid = y_test.loc[valid]
    y_fuzzy_valid = y_fuzzy.loc[valid].astype(int)
    y_ml_valid = pd.Series(y_pred, index=test_indices).loc[valid]

    fuzzy_accuracy = accuracy_score(
        y_true_valid, y_fuzzy_valid
    )
    ml_accuracy = accuracy_score(
        y_test, y_pred
    )

    fuzzy_f1 = f1_score(
        y_true_valid,
        y_fuzzy_valid,
        average="macro",
        zero_division=0,
    )
    ml_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    
    print("СРАВНЕНИЕ НЕЧЁТКОЙ МОДЕЛИ И НЕЙРОННОЙ СЕТИ")
   

    print(f"\nВсего записей: {len(df)}")
    print(f"Тестовых записей: {len(y_test)}")
    print(f"Тестовых записей с результатом нечёткой модели: {valid.sum()}")

    print("\n--- Нечёткая модель ---")
    print(f"Accuracy: {fuzzy_accuracy:.4f}")
    print(f"Macro F1: {fuzzy_f1:.4f}")
    print("\nОтчёт классификации:")
    print(
        classification_report(
            y_true_valid,
            y_fuzzy_valid,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("\n--- Нейронная сеть MLPClassifier ---")
    print(f"Accuracy: {ml_accuracy:.4f}")
    print(f"Macro F1: {ml_f1:.4f}")
    print("\nОтчёт классификации:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # Матрицы ошибок
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ConfusionMatrixDisplay(
        confusion_matrix(
            y_true_valid,
            y_fuzzy_valid,
            labels=[0, 1, 2],
        ),
        display_labels=CLASS_NAMES,
    ).plot(ax=axes[0], colorbar=False, values_format="d")

    axes[0].set_title("Нечёткая модель")
    axes[0].tick_params(axis="x", rotation=20)

    ConfusionMatrixDisplay(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2],
        ),
        display_labels=CLASS_NAMES,
    ).plot(ax=axes[1], colorbar=False, values_format="d")

    axes[1].set_title("Нейронная сеть MLPClassifier")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "model_confusion_matrices.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Диаграмма метрик
    metrics = pd.DataFrame(
        {
            "Модель": ["Нечёткая модель", "MLPClassifier"],
            "Accuracy": [fuzzy_accuracy, ml_accuracy],
            "Macro F1": [fuzzy_f1, ml_f1],
        }
    )

    metrics.to_csv(
        BASE_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.set_index("Модель").plot(
        kind="bar",
        figsize=(9, 5),
        ylim=(0, 1),
        rot=0,
    )

    plt.title("Сравнение качества моделей")
    plt.ylabel("Значение метрики")
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "model_metrics.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("\nМатрицы ошибок сохранены:")
    print(GRAPH_DIR / "model_confusion_matrices.png")
    print("\nГрафик метрик сохранён:")
    print(GRAPH_DIR / "model_metrics.png")
    print("\nТаблица метрик сохранена:")
    print(BASE_DIR / "model_comparison.csv")

    # Пример оценки нового подразделения нейронной сетью
    new_department = pd.DataFrame(
        [{
            "revenue": 15.0,
            "costs": 8.0,
            "employees": 100,
            "productivity": 130.0,
            "overdue_orders": 5,
            "defects": 3,
            "plan_percent": 110.0,
            "profitability": (15.0 - 8.0) / 15.0 * 100,
            "revenue_per_employee": 15.0 / 100,
        }]
    )

    predicted_state = int(model.predict(new_department)[0])

    print("\n--- Пример нового подразделения ---")
    print(f"Прогноз MLPClassifier: {STATE_NAMES[predicted_state]}")


if __name__ == "__main__":
    main()