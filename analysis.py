
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Папка, в которой находятся файлы проекта
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
GRAPHS_DIR = BASE_DIR / "graphs"

# Названия классов для отображения на графиках
STATE_NAMES = {
    0: "Проблемное",
    1: "Требует внимания",
    2: "Устойчивое"
}


def load_data():
    """Загрузка и проверка исходного набора данных."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Файл data.csv не найден. Сначала запустите main.py."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "period", "department", "revenue", "costs",
        "employees", "productivity", "overdue_orders",
        "defects", "plan_percent", "profitability",
        "revenue_per_employee", "state"
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"В данных отсутствуют столбцы: {missing}")

    df["period"] = pd.to_datetime(
        df["period"], format="%Y-%m"
    )

    return df


def show_statistics(df):
    """Вывод основных показателей и проверка качества данных."""
    
    print("АНАЛИЗ ДАННЫХ")
   
    print(f"\nКоличество строк: {len(df)}")
    print(f"Количество столбцов: {len(df.columns)}")
    print(f"Подразделений: {df['department'].nunique()}")
    print(
        f"Период наблюдений: "
        f"{df['period'].min():%Y-%m} — "
        f"{df['period'].max():%Y-%m}"
    )

    print("\nПропущенные значения:")
    print(df.isnull().sum().to_string())

    print("\nКоличество дубликатов:")
    print(df.duplicated().sum())

    print("\nРаспределение состояний:")
    counts = df["state"].value_counts().sort_index()

    for state, count in counts.items():
        name = STATE_NAMES.get(state, "Неизвестный класс")
        percent = count / len(df) * 100
        print(f"{state} — {name}: {count} ({percent:.1f}%)")

    print("\nОписательная статистика:")
    numeric_columns = [
        "revenue", "costs", "employees", "productivity",
        "overdue_orders", "defects", "plan_percent",
        "profitability", "revenue_per_employee"
    ]

    print(df[numeric_columns].describe().round(2).to_string())

    print("\nСредние показатели по состояниям:")
    print(
        df.groupby("state")[
            [
                "revenue", "costs", "profitability",
                "plan_percent", "overdue_orders", "defects"
            ]
        ].mean().round(2).to_string()
    )


def build_graphs(df):
    """Построение и сохранение четырёх аналитических графиков."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # График 1. Распределение состояний
    counts = (
        df["state"]
        .value_counts()
        .reindex([0, 1, 2], fill_value=0)
    )

    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        [STATE_NAMES[i] for i in counts.index],
        counts.values
    )

    plt.title("Распределение состояний подразделений")
    plt.xlabel("Состояние")
    plt.ylabel("Количество наблюдений")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "01_state_distribution.png", dpi=150)
    plt.close()

    # График 2. Распределение рентабельности по классам
    plt.figure(figsize=(9, 5))

    for state in [0, 1, 2]:
        values = df.loc[
            df["state"] == state, "profitability"
        ]
        plt.hist(
            values,
            bins=20,
            alpha=0.5,
            label=STATE_NAMES[state]
        )

    plt.title("Распределение рентабельности по состояниям")
    plt.xlabel("Рентабельность, %")
    plt.ylabel("Количество наблюдений")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "02_profitability.png", dpi=150)
    plt.close()

    # График 3. Корреляционная матрица
    numeric_columns = [
        "revenue", "costs", "employees", "productivity",
        "overdue_orders", "defects", "plan_percent",
        "profitability", "revenue_per_employee", "state"
    ]

    corr = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(numeric_columns)))
    ax.set_yticks(range(len(numeric_columns)))
    ax.set_xticklabels(
        numeric_columns, rotation=45, ha="right"
    )
    ax.set_yticklabels(numeric_columns)

    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7
            )

    ax.set_title("Корреляционная матрица показателей")
    fig.colorbar(image, ax=ax, label="Коэффициент корреляции")
    fig.tight_layout()
    fig.savefig(
        GRAPHS_DIR / "03_correlation_matrix.png",
        dpi=150
    )
    plt.close(fig)

    # График 4. Динамика среднемесячной выручки
    monthly = (
        df.groupby("period", as_index=False)["revenue"]
        .mean()
    )

    plt.figure(figsize=(11, 5))
    plt.plot(
        monthly["period"],
        monthly["revenue"],
        marker="o",
        markersize=2,
        linewidth=1.5
    )

    plt.title("Динамика среднемесячной выручки")
    plt.xlabel("Месяц")
    plt.ylabel("Средняя выручка, млн")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(
        GRAPHS_DIR / "04_revenue_dynamics.png",
        dpi=150
    )
    plt.close()

    print(f"\nЧетыре графика сохранены в папке: {GRAPHS_DIR}")


if __name__ == "__main__":
    data = load_data()
    show_statistics(data)
    build_graphs(data)

    print("\nАнализ успешно завершён!")
