def analyze_PIV_data(velocity_max, shear_stress, heart_rate_bpm):
    results = {}
    results["velocity_max_ms"] = velocity_max
    results["shear_stress_Pa"] = shear_stress
    results["heart_rate"] = heart_rate_bpm
    if velocity_max > 2.0:
        results["velocity_status"] = "HIGH - possible stenosis"
    else:
        results["velocity_status"] = "NORMAL"
    if shear_stress > 10:
        results["shear_status"] = "HIGH - platelet activation risk"
    else:
        results["shear_status"] = "NORMAL"
    return results
