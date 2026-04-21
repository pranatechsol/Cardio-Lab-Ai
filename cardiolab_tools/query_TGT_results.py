def query_TGT_results(TAT,PF12,hemo,plt,t):
    risk=sum([TAT>8,PF12>2.0,hemo>20,plt<150])
    return {"risk":"HIGH" if risk>=3 else "MODERATE" if risk>=2 else "LOW"}
