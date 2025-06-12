import pandas as pd

def preprocess_new_data(df_raw, feature_columns, scaler):
    df = df_raw.rename(columns={
        'Ages': 'Usia',
        'Gender': 'Jenis_Kelamin',
        'Height': 'Tinggi_cm',
        'Weight': 'Berat_kg',
        'Protein': 'Protein_g',
        'Sugar': 'Gula_g',
        'Sodium': 'Sodium_mg',
        'Calories': 'Kalori',
        'Carbohydrates': 'Karbohidrat_g',
        'Fiber': 'Serat_g',
        'Fat': 'Lemak_g'
    })
    numeric_cols = ['Usia','Tinggi_cm','Berat_kg','Protein_g','Gula_g',
                    'Sodium_mg','Kalori','Karbohidrat_g','Serat_g','Lemak_g']
    categorical_cols = ['Jenis_Kelamin']
    missing = [c for c in numeric_cols + categorical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom input hilang: {missing}")
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df
