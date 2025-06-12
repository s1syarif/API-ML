def infer_health_score_custom(
    energi, protein, lemak_total, karbohidrat, serat, gula, garam,
    target_energi, target_protein, target_lemak_total, target_karbohidrat, target_serat, target_gula, target_garam
):
    r_energi      = min(energi      / target_energi,      1.0) if target_energi else 1.0
    r_protein     = min(protein     / target_protein,     1.0) if target_protein else 1.0
    r_lemak_total = min(lemak_total / target_lemak_total, 1.0) if target_lemak_total else 1.0
    r_karbohidrat = min(karbohidrat / target_karbohidrat, 1.0) if target_karbohidrat else 1.0
    r_serat       = min(serat       / target_serat,       1.0) if target_serat else 1.0
    r_gula        = min(gula        / target_gula,        1.0) if target_gula else 1.0
    r_garam       = min(garam       / target_garam,       1.0) if target_garam else 1.0
    avg_ratio = (r_energi + r_protein + r_lemak_total + r_karbohidrat + r_serat + r_gula + r_garam) / 7
    score = int(avg_ratio * 9) + 1
    return max(1, min(score, 10))
