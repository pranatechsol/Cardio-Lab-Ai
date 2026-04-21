def analyze_PIV_data(v,s,hr):
    return {"velocity":"HIGH" if v>2.0 else "NORMAL","shear":"HIGH" if s>10 else "NORMAL"}
