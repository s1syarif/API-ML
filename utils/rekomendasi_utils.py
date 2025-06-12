import pandas as pd
import numpy as np

def calculate_remaining_needs(full_targets, consumed):
    remaining = {}
    for nutrient_key, target_value in full_targets.items():
        consumed_value = consumed.get(nutrient_key, 0)
        remaining[nutrient_key] = max(0, target_value - consumed_value)
    return remaining

def generate_recommendations(df, needs, num_prod=3, iters=30000):
    best_combo, best_score = None, float('inf')
    sisa_gula_limit = needs.get('sugar_g', float('inf'))
    sisa_sodium_limit = needs.get('sodium_mg', float('inf'))
    for _ in range(iters):
        combo = df.sample(n=num_prod) if len(df) >= num_prod else df
        rekom_gula = combo.get('total_sugar_g', pd.Series([0]*num_prod)).sum()
        rekom_sodium = combo.get('total_sodium_mg', pd.Series([0]*num_prod)).sum()
        if rekom_gula > sisa_gula_limit or rekom_sodium > sisa_sodium_limit:
            continue
        nutr = {
            'energy_kal': combo.get('total_energy_kal', pd.Series([0]*num_prod)).sum(),
            'protein_g': combo.get('total_protein_g', pd.Series([0]*num_prod)).sum(),
            'fat_g': combo.get('total_fat_g', pd.Series([0]*num_prod)).sum(),
            'carbohydrate_g': combo.get('total_carbohydrate_g', pd.Series([0]*num_prod)).sum(),
            'fiber_g': combo.get('total_fiber_g', pd.Series([0]*num_prod)).sum()
        }
        scores = []
        for k in ['energy_kal', 'protein_g', 'fat_g', 'carbohydrate_g', 'fiber_g']:
            if needs.get(k, 0) > 0:
                scores.append(abs(nutr[k] - needs[k]) / needs[k])
        score_makro_avg = np.mean(scores) if scores else 0
        score_gula = (rekom_gula / sisa_gula_limit) if sisa_gula_limit > 0 else 0
        score_sod = (rekom_sodium / sisa_sodium_limit) if sisa_sodium_limit > 0 else 0
        score_limit = (score_gula + score_sod) / 2
        total_score = (score_makro_avg * 0.8) + (score_limit * 0.2)
        if total_score < best_score:
            best_score, best_combo = total_score, combo
    return best_combo

def rekomendasi_logic(target_harian, konsumsi, df_model):
    konsumsi_final = {k: konsumsi.get(k, 0) for k in target_harian.keys()}
    sisa_kebutuhan = calculate_remaining_needs(target_harian, konsumsi_final)
    fokus_kurang = [k for k, target in target_harian.items() if target > 0 and konsumsi_final.get(k, 0) < 0.8 * target]
    sisa_kebutuhan_fokus = {k: sisa_kebutuhan[k] for k in fokus_kurang}
    rekomendasi = generate_recommendations(df_model, sisa_kebutuhan_fokus, num_prod=3)
    hasil = []
    if rekomendasi is not None:
        for _, row in rekomendasi.iterrows():
            hasil.append({
                'product_name': row.get('product_name', ''),
                'skor_gizi': row.get('skor_gizi', 0),
                'Energi': f"{row.get('total_energy_kal', 0):.1f} kkal",
                'Protein': f"{row.get('total_protein_g', 0):.1f} g",
                'Lemak total': f"{row.get('total_fat_g', 0):.1f} g",
                'Karbohidrat': f"{row.get('total_carbohydrate_g', 0):.1f} g",
                'Serat': f"{row.get('total_fiber_g', 0):.1f} g",
                'Gula': f"{row.get('total_sugar_g', 0):.1f} g",
                'Garam': f"{row.get('total_sodium_mg', 0):.0f} mg"
            })
    return {'rekomendasi': hasil, 'gizi_fokus': fokus_kurang}
