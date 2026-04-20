def query_TGT_results(TAT, PF12, free_hemoglobin, platelet_count, time_min):
    results = {}
    results["time_minutes"] = time_min
    results["TAT_status"] = "HIGH coagulation" if TAT > 15 else "NORMAL"
    results["PF12_status"] = "HIGH thrombin" if PF12 > 2.0 else "NORMAL"
    results["hemolysis_status"] = "HIGH hemolysis" if free_hemoglobin > 50 else "NORMAL"
    results["platelet_status"] = "LOW platelets" if platelet_count < 150 else "NORMAL"
    risk = sum([TAT>15, PF12>2.0, free_hemoglobin>50, platelet_count<150])
    results["overall"] = "HIGH THROMBOGENIC RISK" if risk>=3 else "MODERATE RISK" if risk>=2 else "LOW RISK"
    return results
