import numpy as np
import pandas as pd

filename = r"D:\Навчання\Технології збору та обробки даних\pw1\2019-Oct_S.csv"
chunk_size = 50_000

print("Крок 1: Єдиний прохід по файлу та збір даних у пам'ять...")

# Словник, де ключем буде product_id, а значенням — список УСІХ його цін
all_products_data = {}
chunk_idx = 0

for chunk in pd.read_csv(filename, chunksize=chunk_size, encoding="utf-8"):
    chunk_idx += 1
    if chunk_idx % 10 == 0:
        print(f"  -> Оброблено {chunk_idx} чанків ({chunk_idx * chunk_size:,} рядків)...")

    chunk.columns = chunk.columns.str.strip().str.lower()

    # Фільтруємо лише покупки
    chunk = chunk[
        (chunk["event_type"] == "purchase")
        & (chunk["price"] > 0)
        & (chunk["product_id"].notna())
    ]

    # Групуємо всередині чанку і перетворюємо ціни в списки
    grouped = chunk.groupby("product_id")["price"].apply(list)

    for prod_id, prices in grouped.items():
        pid = int(prod_id)
        if pid not in all_products_data:
            all_products_data[pid] = []
        all_products_data[pid].extend(prices)

if not all_products_data:
    print("Не знайдено подій з типом 'purchase'.")
    exit()

print("\nРозрахунок метрик та формування звіту...")

# Обчислюємо ВСІ базові метрики для КОЖНОГО товару
full_stats = []
for pid, prices in all_products_data.items():
    arr = np.array(prices)
    full_stats.append({
        "product_id": pid,
        "sum": np.sum(arr),
        "count": len(arr),
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr, ddof=1) if len(arr) > 1 else np.nan
    })

# Перетворюємо у DataFrame та сортуємо за виручкою
df_all = pd.DataFrame(full_stats).sort_values(by="sum", ascending=False).reset_index(drop=True)

# Загальна виручка
total_revenue = df_all["sum"].sum()

# Формуємо ТОП-5 та Інші
top_5 = df_all.head(5).copy()
others_totals = df_all.iloc[5:]

# Створюємо рядок для "Інших"
others_row = pd.DataFrame({
    "product_id": ["Інші (Others)"],
    "count": [others_totals["count"].sum()],
    "sum": [others_totals["sum"].sum()],
    "mean": [np.nan],
    "median": [np.nan],
    "std": [np.nan],
})

# Перейменовуємо для фінального виводу
top_5.rename(columns={"product_id": "product"}, inplace=True)
others_row.rename(columns={"product_id": "product"}, inplace=True)

# Об'єднуємо звіт
final_report = pd.concat([top_5, others_row], ignore_index=True)
final_report["share_of_revenue_%"] = (final_report["sum"] / total_revenue) * 100
final_report = final_report.round(2)

print("\n--- Фінальний звіт (Один прохід: Топ-5 + Інші) ---")
print(final_report.to_string(index=False))