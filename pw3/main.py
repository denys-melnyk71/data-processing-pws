import gc
import pandas as pd

filename = r"D:\Навчання\Технології збору та обробки даних\pw3\PJME_hourly.csv"

# Завантаження оригінального файлу
print("Завантаження датасету...")
df = pd.read_csv(filename)

# Перетворимо Datetime з типу object (текст) у правильний тип datetime64,
# оскільки Pandas за замовчуванням читає дати як важкі текстові рядки.
df["Datetime"] = pd.to_datetime(df["Datetime"])

# Функція оцінки пам'яті
def print_mem_report(dataframe, label):
    print(f"\n--- Пам'ять: {label} ---")
    mem = dataframe.memory_usage(deep=True)
    total = mem.sum() / (1024 ** 2)
    
    types_df = pd.DataFrame({
        "Type": dataframe.dtypes,
        "Memory (MB)": (mem.drop("Index") / (1024 ** 2)).round(2)
    })
    print(types_df)
    print(f"ЗАГАЛОМ У RAM: {total:.2f} MB")
    return total

# Фіксуємо початковий стан
ram_before = print_mem_report(df, "ДО ОПТИМІЗАЦІЇ")

# Оптимізація типів
print("\nОптимізація типів через .astype()...")

# Розбиваємо дату на окремі складові й одразу оптимізуємо їхні типи
df["Year"] = df["Datetime"].dt.year.astype("uint16")
df["Month"] = df["Datetime"].dt.month.astype("uint8")
df["Day"] = df["Datetime"].dt.day.astype("uint8")
df["Hour"] = df["Datetime"].dt.hour.astype("uint8")

# Оптимізуємо тип для колонки PJME_MW
df["PJME_MW"] = df["PJME_MW"].astype("float32")

# Видалення оригінальної колонку Datetime з пам'яті
df.drop(columns=["Datetime"], inplace=True)

# Очищення пам'ять від залишків видаленої колонки
gc.collect()

# Фіксуємо кінцевий стан (вже без Datetime, але з новими оптимізованими колонками)
ram_after = print_mem_report(df, "ПІСЛЯ ОПТИМІЗАЦІЇ")

# Порівняльний аналіз
print("\n --- РЕЗУЛЬТАТ ОПТИМІЗАЦІЇ РЕАЛЬНОГО ДАТАСЕТУ ---")
saved_ram = ram_before - ram_after
percent = (saved_ram / ram_before) * 100

print(f"Початковий об'єм у пам'яті:       {ram_before:.2f} MB")
print(f"Оптимізований об'єм:              {ram_after:.2f} MB")
print(f"Збережено RAM:                    {saved_ram:.2f} MB")
print(f"Ефективність стиснення:           {percent:.2f} %")