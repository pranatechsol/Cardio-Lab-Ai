def analyze_uPAD_image(R, G, B):
    results = {}
    results["RGB"] = {"R": R, "G": G, "B": B}
    orange_score = R - B
    creatinine_mgdL = round(0.02 * orange_score - 0.5, 2)
    creatinine_mgdL = max(0, creatinine_mgdL)
    results["estimated_creatinine_mgdL"] = creatinine_mgdL
    if creatinine_mgdL < 1.2:
        results["CKD_stage"] = "Normal"
    elif creatinine_mgdL < 1.5:
        results["CKD_stage"] = "Borderline"
    elif creatinine_mgdL < 3.0:
        results["CKD_stage"] = "Stage 2 CKD"
    elif creatinine_mgdL < 6.0:
        results["CKD_stage"] = "Stage 3-4 CKD"
    else:
        results["CKD_stage"] = "Stage 5 CKD"
    return results
