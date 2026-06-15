import pandas as pd

# def impute_mean(series):
#     return series.fillna(series.mean())

# def impute_median(series):
#     return series.fillna(series.median())

# def impute_previous(series):
#     return series.ffill().bfill()

# def impute_next(series):
#     return series.bfill().ffill()


def impute_mean(series):
    values = series.copy()

    total = 0
    count = 0

    for value in values:
        if not pd.isna(value):
            total += value
            count += 1

    mean_value = total / count

    for i in range(len(values)):
        if pd.isna(values.iloc[i]):
            values.iloc[i] = mean_value

    return values


def impute_median(series):
    values = series.copy()

    non_nan = []

    for value in values:
        if not pd.isna(value):
            non_nan.append(value)

    non_nan.sort()

    n = len(non_nan)

    if n % 2 == 0:
        median_value = (non_nan[n // 2 - 1] + non_nan[n // 2]) / 2
    else:
        median_value = non_nan[n // 2]

    for i in range(len(values)):
        if pd.isna(values.iloc[i]):
            values.iloc[i] = median_value

    return values


def impute_previous(series):
    values = series.copy()

    last_value = None

    for i in range(len(values)):
        if pd.isna(values.iloc[i]):
            if last_value is not None:
                values.iloc[i] = last_value
        else:
            last_value = values.iloc[i]

    # Якщо пропуски були на початку
    first_valid = None

    for value in values:
        if not pd.isna(value):
            first_valid = value
            break

    for i in range(len(values)):
        if pd.isna(values.iloc[i]):
            values.iloc[i] = first_valid
        else:
            break

    return values


def impute_next(series):
    values = series.copy()

    next_value = None

    for i in range(len(values) - 1, -1, -1):
        if pd.isna(values.iloc[i]):
            if next_value is not None:
                values.iloc[i] = next_value
        else:
            next_value = values.iloc[i]

    # Якщо пропуски були в кінці
    last_valid = None

    for i in range(len(values) - 1, -1, -1):
        if not pd.isna(values.iloc[i]):
            last_valid = values.iloc[i]
            break

    for i in range(len(values) - 1, -1, -1):
        if pd.isna(values.iloc[i]):
            values.iloc[i] = last_valid
        else:
            break

    return values


def fill_data(df_clean, df_corrupted):
    df_corrupted["Mean"] = impute_mean(df_corrupted["PJME_MW"])
    df_corrupted["Median"] = impute_median(df_corrupted["PJME_MW"])
    df_corrupted["Previous"] = impute_previous(df_corrupted["PJME_MW"])
    df_corrupted["Next"] = impute_next(df_corrupted["PJME_MW"])