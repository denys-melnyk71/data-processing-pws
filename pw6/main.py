import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_NORMAL_LUX = 0.33
MAX_NORMAL_LUX = 0.67

dir_path = r"D:\Навчання\Технології збору та обробки даних\pw6"
# Словник із шляхами до твоїх реальних CSV файлів
FILE_PATHS = {
    "LED (Світлодіодна)":           dir_path + r"\led_lamp.csv",
    "Fluorescent (Люмінесцентна)":  dir_path + r"\fluorescent_lamp.csv",
    "Incandescent (Розжарювання)":  dir_path + r"\incandescent_lamp.csv",
}


def load_and_clean_data(file_path, label):
    """Завантажує новий датасет освітлення (Luminance vs Time)."""

    if not os.path.exists(file_path):
        print(f"Файл {file_path} не знайдено. Програму завершено.")
        sys.exit(1)

    df = pd.read_csv(file_path)

    # чистимо назви колонок
    df.columns = [c.strip().lower() for c in df.columns]

    # нові колонки
    time_col = "t"
    value_col = "luminance"

    # час у секундах
    df["seconds"] = df[time_col].astype(float)

    # сигнал
    df["lux"] = df[value_col].astype(float)

    # обрізаємо 60 секунд
    df = df[df["seconds"] <= 60].copy()

    return df[["seconds", "lux"]]


def analyze_lamps(datasets):
    """Проводить статистичний аналіз отриманих з датчика даних."""
    metrics = []

    for label, df in datasets.items():
        mean_lux = df["lux"].mean()
        median_lux = df["lux"].median()
        std_lux = df["lux"].std()
        #max_lux = df["lux"].max()
        #min_lux = df["lux"].min()

        # Коефіцієнт варіації (міра стабільності світла: чим менше, тим краще)
        coef_variation = (std_lux / mean_lux) * 100

        # Перевірка на відповідність нормам для читання (300-500 Лк)
        if mean_lux < MIN_NORMAL_LUX:
            status = "Недостатнє освітлення (Втомлює очі)"
        elif MIN_NORMAL_LUX <= mean_lux <= MAX_NORMAL_LUX:
            status = "Ідеально для читання/роботи"
        else:
            status = "Занадто яскраво (Сліпить очі)"

        metrics.append(
            {
                "Тип лампи": label,
                "Сер. освітленість": round(mean_lux, 2),
                "Медіана": round(median_lux, 2),
                "Мерехтіння (Std Dev)": round(std_lux, 2),
                "Коеф. варіації (%)": round(coef_variation, 2),
                "Ергономіка": status,
            }
        )

    return pd.DataFrame(metrics)


def plot_results(datasets, df_metrics):
    """Будує графіки для звіту."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    # Графік: Хронологічний графік за 1 хвилину
    for (label, df), color in zip(datasets.items(), colors):
        ax1.plot(
            df["seconds"],
            df["lux"],
            label=f"{label} (Середнє: {df['lux'].mean():.1f} Лк)",
            color=color,
            alpha=0.8,
            linewidth=1.5,
        )

    # Підсвічуємо зону санітарної норми зеленим кольором
    ax1.axhspan(
        MIN_NORMAL_LUX,
        MAX_NORMAL_LUX,
        color="green",
        alpha=0.1,
        label="Норма ДСТУ (300-500 Лк)",
    )

    ax1.set_title(
        "Динаміка рівня освітленості протягом 1 хвилини вимірювання", fontsize=12
    )
    ax1.set_xlabel("Час експерименту (секунди)")
    ax1.set_ylabel("Освітленість (Relative luminance))")
    ax1.set_xlim(0, 60)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Діаграма: Порівняння стабільності світлового потоку (Коефіцієнт варіації)
    # Чим більший коефіцієнт — тим сильніше лампа мерехтить або тим нестабільніша напруга
    bars = ax2.barh(
        df_metrics["Тип лампи"],
        df_metrics["Коеф. варіації (%)"],
        color=colors,
        alpha=0.7,
        edgecolor="black",
        height=0.4,
    )
    ax2.bar_label(bars, fmt="%.2f %%", padding=8, fontsize=10, weight="bold")

    ax2.set_title(
        "Коефіцієнт варіації пульсації світла (менше = стабільніше/комфортніше для очей)",
        fontsize=12,
    )
    ax2.set_xlabel("Коефіцієнт варіації (%)")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()


# --- ГОЛОВНИЙ БЛОК ВИКОНАННЯ ---
if __name__ == "__main__":
    print("--- Запуск аналізу датчика освітленості ---")

    # 1. Завантаження та обробка даних
    data_collection = {}
    for lamp_name, path in FILE_PATHS.items():
        data_collection[lamp_name] = load_and_clean_data(path, lamp_name)

    # 2. Розрахунок математичних метрик
    summary_table = analyze_lamps(data_collection)

    # 3. Вивід результатів у консоль для звіту
    print("\n" + "=" * 60 + " СТАТИСТИЧНИЙ ЗВІТ " + "=" * 60)
    print(summary_table.to_string(index=False))
    print("=" * 139)

    # 4. Візуалізація результатів
    plot_results(data_collection, summary_table)