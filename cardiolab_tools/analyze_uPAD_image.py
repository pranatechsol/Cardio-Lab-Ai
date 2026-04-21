def analyze_uPAD_image(R,G,B):
    c=max(0,round(0.018*(R-B)-0.3,2))
    s="Normal" if c<1.2 else "Borderline" if c<1.5 else "Stage2" if c<3.0 else "Stage3-4" if c<6.0 else "Stage5"
    return {"creatinine":c,"stage":s}
