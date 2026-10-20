from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
OUTPUT_PATH = BASE_DIR / "fuzzy_results.csv"


def create_fuzzy_system():
    """Создаёт нечёткую систему оценки состояния подразделения."""

    # Входные переменные
    profitability = ctrl.Antecedent(
        np.arange(-10, 51, 1), "profitability"
    )
    plan_percent = ctrl.Antecedent(
        np.arange(65, 131, 1), "plan_percent"
    )
    overdue_orders = ctrl.Antecedent(
        np.arange(0, 41, 1), "overdue_orders"
    )
    defects = ctrl.Antecedent(
        np.arange(0, 26, 1), "defects"
    )

    # Выходная переменная: оценка от 0 до 100
    health = ctrl.Consequent(
        np.arange(0, 101, 1), "health"
    )

    # Рентабельность
    profitability["low"] = fuzz.trapmf(
        profitability.universe, [-10, -10, 5, 15]
    )
    profitability["medium"] = fuzz.trimf(
        profitability.universe, [5, 20, 35]
    )
    profitability["high"] = fuzz.trapmf(
        profitability.universe, [25, 35, 50, 50]
    )

    # Выполнение плана
    plan_percent["low"] = fuzz.trapmf(
        plan_percent.universe, [65, 65, 80, 95]
    )
    plan_percent["medium"] = fuzz.trimf(
        plan_percent.universe, [85, 100, 115]
    )
    plan_percent["high"] = fuzz.trapmf(
        plan_percent.universe, [105, 115, 130, 130]
    )

    # Просроченные заказы
    overdue_orders["low"] = fuzz.trapmf(
        overdue_orders.universe, [0, 0, 4, 10]
    )
    overdue_orders["medium"] = fuzz.trimf(
        overdue_orders.universe, [5, 15, 25]
    )
    overdue_orders["high"] = fuzz.trapmf(
        overdue_orders.universe, [18, 27, 40, 40]
    )

    # Дефекты
    defects["low"] = fuzz.trapmf(
        defects.universe, [0, 0, 3, 7]
    )
    defects["medium"] = fuzz.trimf(
        defects.universe, [4, 10, 16]
    )
    defects["high"] = fuzz.trapmf(
        defects.universe, [12, 18, 25, 25]
    )

    # Итоговая оценка состояния
    health["problematic"] = fuzz.trapmf(
        health.universe, [0, 0, 20, 40]
    )
    health["attention"] = fuzz.trimf(
        health.universe, [25, 50, 75]
    )
    health["stable"] = fuzz.trapmf(
        health.universe, [60, 80, 100, 100]
    )

    # Правила нечёткого вывода
    rules = [
        # Устойчивое состояние
        ctrl.Rule(
            profitability["high"]
            & plan_percent["high"]
            & overdue_orders["low"]
            & defects["low"],
            health["stable"],
        ),
        ctrl.Rule(
            profitability["high"]
            & plan_percent["high"]
            & overdue_orders["low"],
            health["stable"],
        ),
        ctrl.Rule(
            plan_percent["high"]
            & overdue_orders["low"]
            & defects["low"],
            health["stable"],
        ),
        ctrl.Rule(
            profitability["high"]
            & plan_percent["medium"]
            & overdue_orders["low"]
            & defects["low"],
            health["stable"],
        ),

        # Проблемное состояние
        ctrl.Rule(
            profitability["low"] & overdue_orders["high"],
            health["problematic"],
        ),
        ctrl.Rule(
            profitability["low"] & plan_percent["low"],
            health["problematic"],
        ),
        ctrl.Rule(
            overdue_orders["high"] & defects["high"],
            health["problematic"],
        ),
        ctrl.Rule(
            plan_percent["low"] & defects["high"],
            health["problematic"],
        ),
        ctrl.Rule(
            profitability["low"] & defects["high"],
            health["problematic"],
        ),

        # Состояние требует внимания
        ctrl.Rule(
            profitability["medium"]
            & plan_percent["medium"]
            & overdue_orders["medium"],
            health["attention"],
        ),
        ctrl.Rule(
            profitability["medium"] & plan_percent["low"],
            health["attention"],
        ),
        ctrl.Rule(
            profitability["high"] & plan_percent["low"],
            health["attention"],
        ),
        ctrl.Rule(
            profitability["medium"] & defects["medium"],
            health["attention"],
        ),
        ctrl.Rule(
            plan_percent["high"] & overdue_orders["medium"],
            health["attention"],
        ),
        ctrl.Rule(
            profitability["high"] & defects["high"],
            health["attention"],
        ),
        ctrl.Rule(
            profitability["low"] & plan_percent["high"],
            health["attention"],
        ),
        ctrl.Rule(
            profitability["high"] & overdue_orders["high"],
            health["attention"],
        ),
        ctrl.Rule(
            plan_percent["medium"] & defects["high"],
            health["attention"],
        ),

        # Резервные правила для непокрытых комбинаций.
        # Они помогают системе сформировать результат,
        # даже если ни одно из основных правил не сработало.
        ctrl.Rule(
            profitability["medium"],
            health["attention"],
        ),
        ctrl.Rule(
            profitability["low"],
            health["problematic"],
        ),
        ctrl.Rule(
            profitability["high"],
            health["attention"],
        ),
    ]

    return ctrl.ControlSystem(rules)


def evaluate_department(system, values):
    """Возвращает числовую оценку и категорию состояния."""

    simulation = ctrl.ControlSystemSimulation(system)

    for name, value in values.items():
        simulation.input[name] = float(value)

    simulation.compute()

    score = float(simulation.output["health"])

    if score < 40:
        category = "Проблемное"
    elif score < 70:
        category = "Требует внимания"
    else:
        category = "Устойчивое"

    return round(score, 2), category


def main():
    if not DATA_PATH.exists():
        print(f"Не найден файл данных: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "period",
        "department",
        "profitability",
        "plan_percent",
        "overdue_orders",
        "defects",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("В data.csv отсутствуют столбцы:")
        print(", ".join(missing_columns))
        return

    system = create_fuzzy_system()

    scores = []
    categories = []
    errors = []

    # Оцениваем все записи
    for index, row in df.iterrows():
        values = {
            "profitability": row["profitability"],
            "plan_percent": row["plan_percent"],
            "overdue_orders": row["overdue_orders"],
            "defects": row["defects"],
        }

        try:
            score, category = evaluate_department(system, values)
            scores.append(score)
            categories.append(category)
            errors.append("")

        except (KeyError, ValueError, TypeError) as error:
            scores.append(np.nan)
            categories.append("Ошибка оценки")
            errors.append(str(error))

    df["fuzzy_score"] = scores
    df["fuzzy_state"] = categories
    df["fuzzy_error"] = errors

    # Показываем первые пять результатов
    print("=" * 60)
    print("НЕЧЁТКАЯ ОЦЕНКА СОСТОЯНИЯ ПОДРАЗДЕЛЕНИЙ")
    print("=" * 60)

    for _, row in df.head(5).iterrows():
        print(f"\nПодразделение: {row['department']}")
        print(f"Период: {row['period']}")
        print(f"Оценка состояния: {row['fuzzy_score']}/100")
        print(f"Результат: {row['fuzzy_state']}")

        if row["fuzzy_error"]:
            print(f"Причина ошибки: {row['fuzzy_error']}")

    # Сохраняем результаты
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("РАСПРЕДЕЛЕНИЕ РЕЗУЛЬТАТОВ НЕЧЁТКОЙ МОДЕЛИ")
    print("=" * 60)

    print(df["fuzzy_state"].value_counts())

    error_count = (df["fuzzy_state"] == "Ошибка оценки").sum()
    print(f"\nКоличество ошибок оценки: {error_count}")
    print(f"Всего обработано записей: {len(df)}")
    print(f"Результаты сохранены: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()