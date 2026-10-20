from pathlib import Path
import subprocess
import sys
import pandas as pd

from fuzzy_model import create_fuzzy_system, evaluate_department
from ml_model import FEATURES, TARGET, CLASS_NAMES

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_FILE = Path("data.csv")


def load_data():
    if not DATA_FILE.exists():
        print("Создаём данные...")
        subprocess.run([sys.executable, "main.py"], check=True)

    return pd.read_csv(DATA_FILE)


def show_indicators(df):
    print("\nОСНОВНЫЕ ПОКАЗАТЕЛИ")
    print("-" * 40)
    print("Количество записей:", len(df))
    print("Количество подразделений:", df["department"].nunique())
    print("Количество пропусков:", df.isna().sum().sum())

    columns = [
        "revenue", "costs", "profitability",
        "productivity", "plan_percent"
    ]

    print("\nСредние значения:")
    print(df[columns].mean().round(2))


def show_graphs():
    print("\nСтроим графики...")
    subprocess.run([sys.executable, "analysis.py"], check=True)


def test_fuzzy_model(df):
    print("\nПРОВЕРКА НЕЧЁТКОЙ МОДЕЛИ")

    system = create_fuzzy_system()
    row = df.iloc[0]

    values = {
        "profitability": float(row["profitability"]),
        "plan_percent": float(row["plan_percent"]),
        "overdue_orders": float(row["overdue_orders"]),
        "defects": float(row["defects"])
    }

    score, category = evaluate_department(system, values)

    print("Подразделение:", row["department"])
    print("Рентабельность:", round(values["profitability"], 2))
    print("Оценка модели:", score)
    print("Состояние:", category)


def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def train_neural_network(df):
    print("\nОБУЧЕНИЕ НЕЙРОСЕТИ")

    model, X_test, y_test = train_model(df)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nОтчёт по классам:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    results = X_test.copy()
    results["actual_state"] = y_test
    results["predicted_state"] = predictions
    results["predicted_name"] = [
        CLASS_NAMES[int(value)] for value in predictions
    ]
    results.to_csv("ml_test_results.csv", index=False, encoding="utf-8-sig")

    print("Результаты сохранены в ml_test_results.csv")


def show_metrics(df):
    model, X_test, y_test = train_model(df)
    predictions = model.predict(X_test)

    print("\nМЕТРИКИ НЕЙРОСЕТИ")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

    print("\nPrecision, Recall и F1-score:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    print("Матрица ошибок:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))


def assess_new_department(df):
    print("\nОЦЕНКА НОВОГО ПОДРАЗДЕЛЕНИЯ")

    try:
        revenue = float(input("Выручка (1–20): "))
        costs = float(input("Затраты (0.6–18): "))
        employees = int(input("Количество работников (10–250): "))
        productivity = float(input("Производительность (40–180): "))
        overdue = int(input("Просроченные заказы (0–40): "))
        defects = int(input("Дефекты (0–25): "))
        plan = float(input("Выполнение плана (70–125): "))

        if not (
            1 <= revenue <= 20
            and 0.6 <= costs <= 18
            and 10 <= employees <= 250
            and 40 <= productivity <= 180
            and 0 <= overdue <= 40
            and 0 <= defects <= 25
            and 70 <= plan <= 125
        ):
            print("Ошибка: значения выходят за допустимые пределы.")
            return

        profitability = (revenue - costs) / revenue * 100

        values = {
            "profitability": profitability,
            "plan_percent": plan,
            "overdue_orders": overdue,
            "defects": defects
        }

        fuzzy_system = create_fuzzy_system()
        fuzzy_score, fuzzy_category = evaluate_department(
            fuzzy_system, values
        )

        new_data = pd.DataFrame([{
            "revenue": revenue,
            "costs": costs,
            "employees": employees,
            "productivity": productivity,
            "overdue_orders": overdue,
            "defects": defects,
            "plan_percent": plan,
            "profitability": profitability,
            "revenue_per_employee": revenue / employees
        }])

        model, _, _ = train_model(df)
        prediction = int(model.predict(new_data[FEATURES])[0])
        probabilities = model.predict_proba(new_data[FEATURES])[0]

        print("\nРЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("Рентабельность:", round(profitability, 2), "%")
        print("Нечёткая модель:", fuzzy_category)
        print("Баллы нечёткой модели:", fuzzy_score)
        print("Нейросеть:", CLASS_NAMES[prediction])

        print("\nВероятности классов нейросети:")
        for class_number, probability in zip(model.classes_, probabilities):
            print(f"{CLASS_NAMES[int(class_number)]}: {probability:.1%}")

        fuzzy_class = (
            0 if fuzzy_score < 40
            else 1 if fuzzy_score < 70
            else 2
        )

        if fuzzy_class == prediction:
            print("\nВывод: модели согласны.")
        else:
            print("\nВывод: модели расходятся.")
            print("Рекомендуется дополнительно проверить показатели.")

    except ValueError:
        print("Ошибка: вводи числа.")
    except Exception as error:
        print("Ошибка при оценке:", error)


def compare_models():
    print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
    subprocess.run([sys.executable, "compare_models.py"], check=True)


def main():
    while True:
        print("\n" + "=" * 40)
        print("СИСТЕМА ОЦЕНКИ ПРОИЗВОДСТВЕННЫХ ПОДРАЗДЕЛЕНИЙ")
        print("=" * 40)
        print("1. Сгенерировать или загрузить данные")
        print("2. Показать основные показатели")
        print("3. Построить графики")
        print("4. Проверить нечёткую модель")
        print("5. Обучить нейросеть")
        print("6. Показать метрики и матрицу ошибок")
        print("7. Оценить новое подразделение")
        print("8. Сравнить модели")
        print("0. Выход")

        choice = input("Выбери пункт меню: ").strip()

        try:
            if choice == "1":
                if DATA_FILE.exists():
                    answer = input(
                        "data.csv уже существует. Создать заново? (да/нет): "
                    ).strip().lower()

                    if answer in ("да", "д", "yes", "y"):
                        subprocess.run(
                            [sys.executable, "main.py"], check=True
                        )
                    else:
                        print("Используем существующие данные.")
                else:
                    subprocess.run(
                        [sys.executable, "main.py"], check=True
                    )

            elif choice == "2":
                show_indicators(load_data())

            elif choice == "3":
                load_data()
                show_graphs()

            elif choice == "4":
                test_fuzzy_model(load_data())

            elif choice == "5":
                train_neural_network(load_data())

            elif choice == "6":
                show_metrics(load_data())

            elif choice == "7":
                assess_new_department(load_data())

            elif choice == "8":
                load_data()
                compare_models()

            elif choice == "0":
                print("Программа завершена.")
                break

            else:
                print("Такого пункта нет. Выбери число от 0 до 8.")

        except FileNotFoundError as error:
            print("Не найден файл:", error)
        except subprocess.CalledProcessError:
            print("Не удалось запустить другой файл. Проверь его код.")


if __name__ == "__main__":
    main()
