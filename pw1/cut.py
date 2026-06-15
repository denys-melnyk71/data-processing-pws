import pandas as pd

# Шляхи до файлів
input_filename = r"D:\Навчання\Технології збору та обробки даних\pw1\2019-Oct.csv"
output_filename = (
    r"D:\Навчання\Технології збору та обробки даних\pw1\2019-Oct_S.csv"
)

# Розмір шматка для читання
chunk_size = 100_000

print("Початок створення зменшеної копії...")

# Лічильник для відстеження прогресу
chunk_idx = 0
first_chunk = True

# Читаємо оригінальний гігабайтний файл порціями
for chunk in pd.read_csv(input_filename, chunksize=chunk_size, encoding="utf-8"):
    chunk_idx += 1

    # Відбираємо випадкові 10% рядків із кожного шматка
    chunk_mini = chunk.sample(frac=0.2, random_state=42)

    # Записуємо відібрані рядки у новий файл
    if first_chunk:
        # Для першого шматка створюємо файл і записуємо заголовок (columns)
        chunk_mini.to_csv(output_filename, index=False, mode="w", encoding="utf-8")
        first_chunk = False
    else:
        # Для всіх наступних шматків просто дописуємо дані в кінець (mode='a' - append)
        chunk_mini.to_csv(
            output_filename, index=False, mode="a", header=False, encoding="utf-8"
        )

    # Виводимо прогрес у консоль
    if chunk_idx % 5 == 0:
        print(
            f"  [Прогрес] Оброблено {chunk_idx} великих чанків оригінального файлу..."
        )

print(f"\nГотово! Зменшений файл збережено за шляхом:\n{output_filename}")