
import numpy as np
import pandas as pd
from pathlib import Path


def generate_data(seed=42):
    """
    Создание учебного набора данных о производственных подразделениях.

    Целевая переменная state:
    0 — проблемное состояние;
    1 — требует внимания;
    2 — устойчивое состояние.

    Класс формируется по учебной экспертной формуле,
    учитывающей экономические показатели и операционные риски.
    """

    rng = np.random.default_rng(seed)

    # 20 подразделений и 60 месяцев = 1200 наблюдений
    departments = [f"Цех {i:02d}" for i in range(1, 21)]
    periods = pd.date_range(
        start="2022-01-01",
        end="2026-12-01",
        freq="MS"
    )

    # Формируем все сочетания подразделений и месяцев
    df = pd.MultiIndex.from_product(
        [departments, periods],
        names=["department", "period"]
    ).to_frame(index=False)

    n = len(df)

    # Исходные показатели
    df["revenue"] = np.round(
        rng.uniform(1, 20, n), 2
    )

    # Затраты связаны с выручкой, но имеют случайные отклонения
    cost_ratio = rng.uniform(0.55, 1.10, n)
    df["costs"] = np.round(
        np.clip(df["revenue"] * cost_ratio, 0.6, 18), 2
    )

    df["employees"] = rng.integers(10, 251, n)
    df["productivity"] = np.round(
        rng.uniform(40, 180, n), 2
    )
    df["overdue_orders"] = rng.integers(0, 41, n)
    df["defects"] = rng.integers(0, 26, n)
    df["plan_percent"] = np.round(
        rng.uniform(70, 125, n), 2
    )

    # Расчётные экономические показатели
    df["profitability"] = np.round(
        (df["revenue"] - df["costs"])
        / df["revenue"] * 100,
        2
    )

    df["revenue_per_employee"] = np.round(
        df["revenue"] / df["employees"],
        4
    )

    # Учебная экспертная формула для целевого класса.
    # Положительные баллы означают благоприятные условия,
    # отрицательные — экономические проблемы и операционные риски.
    score = np.zeros(n, dtype=int)

    # Рентабельность
    score += np.select(
        [
            df["profitability"] < 5,
            df["profitability"] < 15,
            df["profitability"] >= 25
        ],
        [-2, -1, 2],
        default=1
    )

    # Выполнение производственного плана
    score += np.select(
        [
            df["plan_percent"] < 85,
            df["plan_percent"] < 95,
            df["plan_percent"] >= 105
        ],
        [-2, -1, 2],
        default=1
    )

    # Просроченные заказы
    score += np.select(
        [
            df["overdue_orders"] >= 25,
            df["overdue_orders"] >= 12,
            df["overdue_orders"] <= 4
        ],
        [-2, -1, 1],
        default=0
    )

    # Дефекты и рекламации
    score += np.select(
        [
            df["defects"] >= 16,
            df["defects"] >= 8,
            df["defects"] <= 3
        ],
        [-2, -1, 1],
        default=0
    )

    # Перевод экспертного балла в один из трёх классов
    df["state"] = np.select(
        [score <= -2, score >= 3],
        [0, 2],
        default=1
    ).astype(int)

    # Приводим дату к удобному для CSV формату
    df["period"] = df["period"].dt.strftime("%Y-%m")

    # Порядок столбцов согласно заданию
    columns = [
        "period",
        "department",
        "revenue",
        "costs",
        "employees",
        "productivity",
        "overdue_orders",
        "defects",
        "plan_percent",
        "profitability",
        "revenue_per_employee",
        "state"
    ]

    df = df[columns]

    # Сохраняем файл рядом с main.py
    output_path = Path(__file__).resolve().parent / "data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nДанные успешно сгенерированы!")
    print(f"Файл: {output_path}")
    print(f"Количество наблюдений: {len(df)}")
    print(f"Количество подразделений: {df['department'].nunique()}")
    print(f"Период: {df['period'].min()} — {df['period'].max()}")

    print("\nПервые 5 строк:")
    print(df.head().to_string(index=False))

    print("\nРаспределение классов состояния:")
    print(
        df["state"]
        .value_counts()
        .sort_index()
        .rename(index={
            0: "0 — Проблемное",
            1: "1 — Требует внимания",
            2: "2 — Устойчивое"
        })
        .to_string()
    )

    print("\nОсновная статистика:")
    print(
        df[
            [
                "revenue",
                "costs",
                "profitability",
                "plan_percent",
                "overdue_orders",
                "defects"
            ]
        ].describe().round(2)
    )

    return df


if __name__ == "__main__":
    generate_data(seed=42)
