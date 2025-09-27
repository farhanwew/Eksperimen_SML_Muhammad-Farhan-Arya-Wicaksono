import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset dari file CSV.
    """
    df = pd.read_csv(path)
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus outlier dengan metode IQR.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[~(
        (df < (Q1 - 1.5 * IQR)) |
        (df > (Q3 + 1.5 * IQR))
    ).any(axis=1)]
    return df_clean

def preprocess_dataset(path: str, output_path: str) -> pd.DataFrame:
    """
    Preprocessing otomatis:
    1. Load dataset
    2. Drop NA & duplicate
    3. Remove outliers (IQR)
    4. Scaling fitur numerik
    5. Save hasil ke CSV
    """
    df = load_dataset(path)

    # Step 1: Drop missing values & duplicates
    df = df.dropna()
    df = df.drop_duplicates()

    # Step 2: Remove outliers
    df = remove_outliers(df)

    # Step 3: Scaling fitur numerik
    num_cols = ["square_feet", "num_rooms", "age", "distance_to_city(km)"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save hasil
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved at: {output_path}")

    return df

if __name__ == "__main__":
    # Contoh pemanggilan
    input_file = "house_prices_dataset.csv"
    output_file = "preprocessing/house_prices_preprocessed.csv"
    preprocess_dataset(input_file, output_file)
