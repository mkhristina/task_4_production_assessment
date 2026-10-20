from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"

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


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Масштабирование признаков и нейронная сеть
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

    # Обучение модели
    model.fit(X_train, y_train)

    # Предсказания на тестовых данных
    y_pred = model.predict(X_test)

    print("=" * 60)
    print("ОЦЕНКА НЕЙРОННОЙ СЕТИ MLPCLASSIFIER")
    print("=" * 60)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочность (Accuracy): {accuracy:.4f}")

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

    print("Матрица ошибок:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2],
        )
    )

    # Сохраняем прогнозы тестовой выборки
    results = X_test.copy()
    results["actual_state"] = y_test
    results["predicted_state"] = y_pred
    results["predicted_name"] = [
        CLASS_NAMES[int(value)] for value in y_pred
    ]

    output_path = BASE_DIR / "ml_test_results.csv"
    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nРезультаты сохранены: {output_path}")


if __name__ == "__main__":
    main()