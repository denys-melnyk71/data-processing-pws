import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from data_filler import fill_data

FILENAME = r"D:\Навчання\Технології збору та обробки даних\pw3\PJME_hourly.csv"

ROWS_TO_DROP = 20         # Скільки рядків поспіль видаляєти
PERIOD = 250         # Період вікна (кожні 500 годин)
ROWS_TO_READ = 1000  #скільки рядків зчитувати

POLY_ORDER = 2   # Порядок поліноміальної інтерполяції
SPLINE_ORDER = 3 # Порядок сплайн-інтерполяції (3 = кубічний сплайн)


def prepare_data():
    # Завантаження та підготовка базової суцільної сітки
    print("Завантаження датасету...")
    df = pd.read_csv(FILENAME)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Очищення від дублікатів та створення ідеальної часової сітки
    duplicated_count = df.duplicated(subset=["Datetime"]).sum()
    if duplicated_count > 0:
        df = df.groupby("Datetime", as_index=False)["PJME_MW"].mean()

    ideal_timeline = pd.date_range(
        start=df["Datetime"].min(), end=df["Datetime"].max(), freq="h"
    )
    df.set_index("Datetime", inplace=True)
    df = df.reindex(ideal_timeline)
    df.index.name = "Datetime"
    df.reset_index(inplace=True)

    # Заповнюємо початкові рідні пропуски для створення еталону
    df["PJME_MW"] = df["PJME_MW"].interpolate(method="linear")

    # Беремо піддослідну вибірку 
    df_clean = df.head(ROWS_TO_READ).copy()

    # Генерація пропусків 
    df_corrupted = df_clean.copy()

    # Генеруємо маску для видалення даних пакетно
    drop_indices = []
    for start_window in range(PERIOD, len(df_corrupted) - ROWS_TO_DROP, PERIOD):
        # Додаємо ROWS_TO_DROP індексів підряд всередині поточного вікна
        end_window = start_window + ROWS_TO_DROP
        drop_indices.extend(range(start_window, end_window))

    drop_indices = np.array(drop_indices)

    # Записуємо NaN у ці позиції
    df_corrupted.loc[drop_indices, "PJME_MW"] = np.nan
    print(f"\nВидалення {ROWS_TO_DROP} рядків кожні {PERIOD} рядків.")
    print(f"Створено {len(drop_indices)} годин відсутності даних.")

    return df_clean, df_corrupted, drop_indices


def error_estimation(df_clean, df_corrupted, drop_indices):
    # Оцінка похибок (Розрахунок RMSE)
    y_true = df_clean.loc[drop_indices, "PJME_MW"]

    mape_mean = mean_absolute_percentage_error(
        y_true,
        df_corrupted.loc[drop_indices, "Mean"]
    ) * 100

    mape_median = mean_absolute_percentage_error(
        y_true,
        df_corrupted.loc[drop_indices, "Median"]
    ) * 100

    mape_previous = mean_absolute_percentage_error(
        y_true,
        df_corrupted.loc[drop_indices, "Previous"]
    ) * 100

    mape_next = mean_absolute_percentage_error(
        y_true,
        df_corrupted.loc[drop_indices, "Next"]
    ) * 100

    print(" --- ПОРІВНЯЛЬНИЙ АНАЛІЗ ПОХИБОК (MAPE) ---")
    print(f"1. Середнє значення:      {mape_mean:.2f} %")
    print(f"2. Медіана:              {mape_median:.2f} %")
    print(f"3. Попереднє значення:   {mape_previous:.2f} %")
    print(f"4. Наступне значення:    {mape_next:.2f} %")
    return mape_mean, mape_median, mape_previous, mape_next


def display_resalts(df_clean, df_corrupted, mape_mean, mape_median, mape_previous, mape_next):
    # Візуалізація результатів
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Графік 1: Порівняння ліній
    ax1.plot(
        df_clean["Datetime"], df_clean["PJME_MW"],
        label="Оригінал (Істинні дані)", color="black", linewidth=2, alpha=0.8, zorder=1
    )

    ax1.plot(
        df_corrupted["Datetime"],
        df_corrupted["Mean"],
        label=f"Середнє ({mape_mean:.1f}%)",
        linestyle="--"
    )

    ax1.plot(
        df_corrupted["Datetime"],
        df_corrupted["Median"],
        label=f"Медіана ({mape_median:.1f}%)",
        linestyle="-."
    )

    ax1.plot(
        df_corrupted["Datetime"],
        df_corrupted["Previous"],
        label=f"Попереднє ({mape_previous:.1f}%)",
        linestyle=":"
    )

    ax1.plot(
        df_corrupted["Datetime"],
        df_corrupted["Next"],
        label=f"Наступне ({mape_next:.1f}%)",
        linestyle=(0, (3, 1, 1, 1))
    )

    # Виділимо зони пропусків вертикальними лініями
    for start_window in range(PERIOD, len(df_corrupted) - ROWS_TO_DROP, PERIOD):
        ax1.axvspan(
            df_clean.loc[start_window, "Datetime"], 
            df_clean.loc[start_window + ROWS_TO_DROP, "Datetime"], 
            color='crimson', alpha=0.2, label="Зона блекауту (NaN)" if start_window == PERIOD else ""
        )

    ax1.set_title(f"Відновлення великих пакетних пропусків (Видалено по {ROWS_TO_DROP} рядків кожні {PERIOD} годин)", fontsize=12)
    ax1.set_xlabel("Дата та час")
    ax1.set_ylabel("Споживання енергії (PJME_MW)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Графік 2: Гістограма похибок у відсотках
    methods = ["Mean", "Median", "Previous value", "Next value"]

    errors = [mape_mean, mape_median, mape_previous, mape_next]

    colors = ["blue", "green", "red", "orange"]

    bars = ax2.barh(methods, errors, color=colors, alpha=0.7, edgecolor="black", height=0.5)
    ax2.bar_label(bars, fmt="%.2f %%", padding=8, fontsize=10, weight="bold")

    ax2.set_title("Величина похибки MAPE методів (менше = краще)", fontsize=12)
    ax2.set_xlabel("Похибка MAPE (%)")
    ax2.set_xlim(0, max(errors) * 1.2)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    df_clean, df_corrupted, drop_indices = prepare_data()
    fill_data(df_clean, df_corrupted)
    mape_mean, mape_median, mape_previous, mape_next = error_estimation(
        df_clean, 
        df_corrupted, 
        drop_indices
    )
    display_resalts(df_clean, df_corrupted, mape_mean, mape_median, mape_previous, mape_next)