import gradio as gr
import os
import requests
from groq import Groq

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

KNOWHOW = """SJSU CardioLab:
MCL: Sylgard 184 PDMS 10:1 ratio 48hr cure green laser PIV 70bpm 5L/min.
TGT: Arduino Uno Stepper Motor 150mL blood sampled at 0 20 40 60min measures TAT PF1.2 hemolysis platelets.
uPAD: Jaffe reaction creatinine plus picric acid gives orange-red color normal 0.6-1.2 mg/dL CKD above 1.5.
MHV: 27mm SJM Regent bileaflet also trileaflet monoleaflet pediatric.
Equipment: Heska HT5 hematology analyzer time-resolved PIV Tygon tubing Arduino Uno."""

def get_pubmed(query):
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":query,"retmax":3,"retmode":"json","sort":"date"},timeout=8)
        ids = r.json()["esearchresult"]["idlist"]
        if not ids: return ""
        return "\n\nPubMed: "+" | ".join(["https://pubmed.ncbi.nlm.nih.gov/"+i for i in ids])
    except: return ""

def respond(message, history):
    if not GROQ_KEY:
        history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":"Error: Add GROQ_API_KEY to Space Settings Secrets tab."})
        return "", history
    try:
        client = Groq(api_key=GROQ_KEY)
        msgs = [{"role":"system","content":"You are CardioLab AI from SJSU Biomedical Engineering built on Biomni Stanford. Expert in MHV MCL PIV TGT uPAD CKD FSI. Remember full conversation. Never invent URLs.\n\n"+KNOWHOW}]
        for item in history:
            if isinstance(item, dict):
                msgs.append({"role":item["role"],"content":item["content"]})
        msgs.append({"role":"user","content":message})
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=600)
        answer = resp.choices[0].message.content + get_pubmed(message)
        history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":answer})
        return "", history
    except Exception as e:
        history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":"Error: "+str(e)})
        return "", history

def piv_tool(velocity, shear, hr):
    v = "HIGH-stenosis" if float(velocity)>2.0 else "NORMAL"
    s = "HIGH-thrombosis" if float(shear)>10 else "ELEVATED" if float(shear)>5 else "NORMAL"
    return "Velocity:"+str(velocity)+"m/s - "+v+"\nShear:"+str(shear)+"Pa - "+s+"\nHR:"+str(hr)+"bpm"

def tgt_tool(tat,pf12,hemo,platelets,time):
    risk=sum([float(tat)>15,float(pf12)>2.0,float(hemo)>50,float(platelets)<150])
    r="HIGH RISK" if risk>=3 else "MODERATE" if risk>=2 else "LOW RISK"
    return "TAT:"+str(tat)+" PF1.2:"+str(pf12)+"\nHemo:"+str(hemo)+" Plt:"+str(platelets)+"\nResult:"+r

def upad_tool(r,g,b):
    c=max(0,round(0.02*(float(r)-float(b))-0.5,2))
    s="Normal" if c<1.2 else "Borderline" if c<1.5 else "Stage2CKD" if c<3.0 else "Stage3-4" if c<6.0 else "Stage5"
    return "Creatinine:"+str(c)+"mg/dL\nStage:"+s

with gr.Blocks(title="CardioLab AI SJSU") as demo:
    gr.Markdown("# CardioLab AI Agent")
    gr.Markdown("### SJSU Biomedical Engineering | Biomni Stanford + Llama 70B + PubMed")
    gr.Markdown("github.com/pranatechsol/Cardio-Lab-Ai")
    with gr.Tab("Research Chat"):
        gr.Markdown("### Chat like ChatGPT for CardioLab research")
        chatbot = gr.Chatbot(height=400, label="CardioLab AI")
        msg_box = gr.Textbox(placeholder="Ask anything about CardioLab...", label="Your message", lines=2)
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary", scale=3)
            clear_btn = gr.Button("Clear", scale=1)
        send_btn.click(respond, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])
        msg_box.submit(respond, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])
    with gr.Tab("PIV Analysis"):
        v=gr.Number(label="Max Velocity m/s",value=1.8)
        s=gr.Number(label="Shear Stress Pa",value=6.5)
        h=gr.Number(label="Heart Rate bpm",value=72)
        out=gr.Textbox(label="Result",lines=4)
        gr.Button("Analyze PIV").click(piv_tool,inputs=[v,s,h],outputs=out)
    with gr.Tab("TGT Results"):
        t1=gr.Number(label="TAT",value=18)
        t2=gr.Number(label="PF1.2",value=2.5)
        t3=gr.Number(label="Free Hemoglobin",value=60)
        t4=gr.Number(label="Platelet Count",value=140)
        t5=gr.Number(label="Time minutes",value=40)
        out2=gr.Textbox(label="Result",lines=5)
        gr.Button("Analyze TGT").click(tgt_tool,inputs=[t1,t2,t3,t4,t5],outputs=out2)
    with gr.Tab("uPAD CKD"):
        r=gr.Number(label="R value",value=210)
        g=gr.Number(label="G value",value=140)
        b=gr.Number(label="B value",value=80)
        out3=gr.Textbox(label="Result",lines=4)
        gr.Button("Analyze uPAD").click(upad_tool,inputs=[r,g,b],outputs=out3)

demo.launch()
