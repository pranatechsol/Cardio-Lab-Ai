import gradio as gr
import os, requests, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from groq import Groq
from PIL import Image

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

KNOWHOW = ("MCL: Sylgard 184 PDMS 10:1 ratio 48hr cure green laser PIV 70bpm 5L/min. "
    "TGT: Arduino Uno Stepper Motor 150mL blood sampled at 0 20 40 60min measures TAT PF1.2 hemolysis platelets. "
    "uPAD: Jaffe reaction creatinine plus picric acid gives orange-red color normal 0.6-1.2 mg/dL CKD above 1.5. "
    "MHV: 27mm SJM Regent bileaflet also trileaflet monoleaflet pediatric.")

CSS = """
body, .gradio-container { background: #f0f4f8 !important; }
.tab-nav { background: #ffffff !important; border-bottom: 2px solid #e2e8f0 !important; padding: 4px 5px 0 5px !important; display: flex !important; flex-wrap: wrap !important; gap: 3px !important; overflow: visible !important; }
.tab-nav button { background: #f7fafc !important; color: #2d3748 !important; border: 1px solid #e2e8f0 !important; border-radius: 6px 6px 0 0 !important; padding: 8px 10px !important; font-weight: 600 !important; font-size: 0.8em !important; margin: 0 !important; white-space: nowrap !important; min-width: 0 !important; }
.tab-nav button:hover { background: #ebf4ff !important; color: #1a237e !important; }
.tab-nav button.selected { background: linear-gradient(135deg, #e63946, #c1121f) !important; color: #ffffff !important; font-weight: 700 !important; }
button.primary { background: linear-gradient(135deg, #e63946 0%, #c1121f 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; }
button.secondary { background: #edf2f7 !important; color: #4a5568 !important; border: 1px solid #cbd5e0 !important; border-radius: 8px !important; }
textarea, input[type=number] { background: #f7fafc !important; color: #1a202c !important; border: 1px solid #cbd5e0 !important; border-radius: 8px !important; }
.message.user { background: linear-gradient(135deg, #e63946, #c1121f) !important; color: white !important; }
.message.bot { background: #ebf4ff !important; color: #1a202c !important; border: 1px solid #bee3f8 !important; }
label span { color: #2b6cb0 !important; font-weight: 600 !important; font-size: 0.85em !important; text-transform: uppercase !important; }
"""

def get_pubmed(query, n=5):
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":query+" AND (mechanical heart valve OR microfluidic OR CKD OR thrombogenicity)","retmax":n,"retmode":"json","sort":"date"},timeout=10)
        ids = r.json()["esearchresult"]["idlist"]
        if not ids: return ""
        return chr(10).join(["https://pubmed.ncbi.nlm.nih.gov/"+i for i in ids])
    except: return ""

def quick_search(query):
    if not query.strip(): return "Please enter a topic."
    pubmed = get_pubmed(query, n=8)
    try:
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query":query+" biomedical","limit":5,"fields":"title,year,url,citationCount"},timeout=10)
        papers = r.json().get("data",[])
        scholar = chr(10).join([p.get("title","")[:80]+" ("+str(p.get("year",""))+")"+chr(10)+"   "+p.get("url","") for p in papers if p.get("url","")])
    except: scholar = ""
    return "PUBMED:"+chr(10)+pubmed+chr(10)+chr(10)+"SCHOLAR:"+chr(10)+scholar

def research_chat(message, history):
    if not GROQ_KEY:
        history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":"Error: Add GROQ_API_KEY to Space Settings."})
        return "", history
    try:
        client = Groq(api_key=GROQ_KEY)
        msgs = [{"role":"system","content":"You are CardioLab AI. Expert in MHV MCL PIV TGT uPAD CKD FSI. Never invent URLs. "+KNOWHOW}]
        for item in history:
            if isinstance(item, dict): msgs.append({"role":item["role"],"content":item["content"]})
        msgs.append({"role":"user","content":message})
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=700)
        answer = resp.choices[0].message.content
        pubmed = get_pubmed(message, n=3)
        if pubmed: answer += chr(10)+chr(10)+"PUBMED:"+chr(10)+pubmed
        history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":answer})
        return "", history
    except Exception as e:
        history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":"Error: "+str(e)})
        return "", history

def voice_chat(audio, history):
    if audio is None:
        history.append({"role":"assistant","content":"Please record your question first."})
        return history
    try:
        client = Groq(api_key=GROQ_KEY)
        with open(audio, "rb") as f:
            tx = client.audio.transcriptions.create(file=("audio.wav", f, "audio/wav"), model="whisper-large-v3")
        msgs = [{"role":"system","content":"You are CardioLab AI. "+KNOWHOW}]
        for item in history:
            if isinstance(item, dict): msgs.append({"role":item["role"],"content":item["content"]})
        msgs.append({"role":"user","content":tx.text})
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=500)
        history.append({"role":"user","content":"[Voice] "+tx.text})
        history.append({"role":"assistant","content":resp.choices[0].message.content})
        return history
    except Exception as e:
        history.append({"role":"assistant","content":"Voice error: "+str(e)})
        return history

def analyze_piv_csv(file):
    if file is None:
        return None, "Please upload a PIV CSV file first."
    try:
        df = pd.read_csv(file.name)
        cols = [c.lower().strip() for c in df.columns]
        df.columns = cols
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None, "No numeric columns found. Check your CSV file."

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#0d1b3e")
        fig.suptitle("PIV Data Analysis — SJSU CardioLab MCL", color="white", fontsize=16, fontweight="bold")

        def style_ax(ax, title, ylabel):
            ax.set_facecolor("#1a2744")
            ax.set_title(title, color="white", fontweight="bold")
            ax.set_ylabel(ylabel, color="#a8b2d8")
            ax.tick_params(colors="#a8b2d8")
            ax.grid(True, alpha=0.2, color="#2d4a8a")
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            for spine in ["bottom","left"]: ax.spines[spine].set_color("#2d4a8a")

        x = range(len(df))
        vel_col = next((c for c in cols if any(k in c for k in ["vel","speed","u","v_mag"])), num_cols[0] if num_cols else None)
        shear_col = next((c for c in cols if any(k in c for k in ["shear","stress","tau","wss"])), num_cols[1] if len(num_cols)>1 else None)

        # Plot 1 - Velocity
        ax1 = axes[0,0]
        if vel_col:
            ax1.plot(df[vel_col], color="#e63946", linewidth=2.5, marker="o", markersize=4)
            ax1.axhline(y=2.0, color="#ffd700", linestyle="--", linewidth=1.5, label="Risk (2.0 m/s)")
            ax1.fill_between(x, df[vel_col], alpha=0.2, color="#e63946")
            ax1.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        style_ax(ax1, "Velocity Profile", "Velocity (m/s)")

        # Plot 2 - Shear
        ax2 = axes[0,1]
        if shear_col:
            ax2.plot(df[shear_col], color="#4361ee", linewidth=2.5, marker="s", markersize=4)
            ax2.axhline(y=5, color="#ffd700", linestyle="--", linewidth=1.5, label="Caution (5 Pa)")
            ax2.axhline(y=10, color="#e63946", linestyle="--", linewidth=1.5, label="High risk (10 Pa)")
            ax2.fill_between(x, df[shear_col], alpha=0.2, color="#4361ee")
            ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        elif len(num_cols)>1:
            ax2.plot(df[num_cols[1]], color="#4361ee", linewidth=2.5)
        style_ax(ax2, "Shear Stress", "Shear Stress (Pa)")

        # Plot 3 - Distribution
        ax3 = axes[1,0]
        if vel_col:
            ax3.hist(df[vel_col].dropna(), bins=20, color="#2ecc71", alpha=0.8, edgecolor="#0d1b3e")
        style_ax(ax3, "Velocity Distribution", "Count")
        ax3.set_xlabel("Value", color="#a8b2d8")

        # Plot 4 - Stats
        ax4 = axes[1,1]
        ax4.set_facecolor("#1a2744")
        ax4.axis("off")
        stats = ""
        risk = []
        for col in num_cols[:3]:
            mn = df[col].mean()
            mx = df[col].max()
            stats += col[:12]+":"+chr(10)+"  Mean: "+str(round(mn,3))+chr(10)+"  Max:  "+str(round(mx,3))+chr(10)+chr(10)
            if "vel" in col and mx > 2.0: risk.append("HIGH VELOCITY: stenosis risk")
            if "shear" in col and mx > 10: risk.append("HIGH SHEAR: thrombosis risk")
        if risk: stats += "RISK FLAGS:"+chr(10)+"  "+chr(10)+"  ".join(risk)
        ax4.text(0.05, 0.95, "SUMMARY STATS"+chr(10)+"━"*18+chr(10)+stats, transform=ax4.transAxes,
            color="white", fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#0d1b3e", edgecolor="#4361ee"))

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=120)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        ai_text = ""
        if GROQ_KEY:
            try:
                client = Groq(api_key=GROQ_KEY)
                msgs = [{"role":"system","content":"You are a PIV expert for SJSU CardioLab. Analyze PIV statistics and give clinical interpretation about velocity, shear stress, stenosis and thrombosis risk."}]
                msgs.append({"role":"user","content":"PIV data stats from 27mm SJM Regent MHV at 70bpm 5L/min:"+chr(10)+df.describe().to_string()[:800]})
                resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=350)
                ai_text = chr(10)+"━"*25+chr(10)+"AI ANALYSIS:"+chr(10)+resp.choices[0].message.content
            except: pass

        return img, "PIV CSV LOADED: "+str(len(df))+" rows, "+str(len(df.columns))+" columns"+chr(10)+"Columns: "+", ".join(df.columns.tolist())+ai_text
    except Exception as e:
        return None, "Error: "+str(e)

def analyze_tgt_csv(file):
    if file is None:
        return None, "Please upload a TGT CSV file first."
    try:
        df = pd.read_csv(file.name)
        cols = [c.lower().strip() for c in df.columns]
        df.columns = cols
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None, "No numeric columns found."

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#0d1b3e")
        fig.suptitle("TGT Blood Analysis — SJSU CardioLab", color="white", fontsize=16, fontweight="bold")

        time_col = next((c for c in cols if "time" in c or "min" in c), None)
        tat_col = next((c for c in cols if "tat" in c or "thrombin" in c), num_cols[0] if num_cols else None)
        pf_col = next((c for c in cols if "pf" in c or "prothrombin" in c), num_cols[1] if len(num_cols)>1 else None)
        hemo_col = next((c for c in cols if "hemo" in c or "hgb" in c), num_cols[2] if len(num_cols)>2 else None)
        plt_col = next((c for c in cols if "platelet" in c or "plt" in c), num_cols[3] if len(num_cols)>3 else None)
        x = df[time_col] if time_col else range(len(df))
        xl = time_col if time_col else "Sample"

        def style_ax(ax, title, ylabel):
            ax.set_facecolor("#1a2744")
            ax.set_title(title, color="white", fontweight="bold")
            ax.set_ylabel(ylabel, color="#a8b2d8")
            ax.set_xlabel(xl, color="#a8b2d8")
            ax.tick_params(colors="#a8b2d8")
            ax.grid(True, alpha=0.2, color="#2d4a8a")
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            for spine in ["bottom","left"]: ax.spines[spine].set_color("#2d4a8a")

        ax1 = axes[0,0]
        if tat_col:
            ax1.plot(x, df[tat_col], color="#e63946", linewidth=2.5, marker="o", markersize=6)
            ax1.axhline(y=8, color="#ffd700", linestyle="--", linewidth=1.5, label="Normal (8 ng/mL)")
            ax1.fill_between(x, df[tat_col], alpha=0.3, color="#e63946")
            ax1.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        style_ax(ax1, "TAT (Thrombin-Antithrombin)", "ng/mL")

        ax2 = axes[0,1]
        if pf_col:
            ax2.plot(x, df[pf_col], color="#4361ee", linewidth=2.5, marker="s", markersize=6)
            ax2.axhline(y=2.0, color="#ffd700", linestyle="--", linewidth=1.5, label="Normal (2.0)")
            ax2.fill_between(x, df[pf_col], alpha=0.3, color="#4361ee")
            ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        style_ax(ax2, "PF1.2 (Prothrombin Fragment)", "nmol/L")

        ax3 = axes[1,0]
        if hemo_col:
            ax3.bar(range(len(df)), df[hemo_col], color="#2ecc71", alpha=0.85, edgecolor="#0d1b3e")
            ax3.axhline(y=20, color="#ffd700", linestyle="--", linewidth=1.5, label="Normal (20 mg/L)")
            ax3.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        style_ax(ax3, "Free Hemoglobin (Hemolysis)", "mg/L")

        ax4 = axes[1,1]
        if plt_col:
            ax4.plot(x, df[plt_col], color="#e67e22", linewidth=2.5, marker="^", markersize=6)
            ax4.axhline(y=150, color="#ffd700", linestyle="--", linewidth=1.5, label="Normal min (150)")
            ax4.fill_between(x, df[plt_col], 150, where=df[plt_col]<150, alpha=0.3, color="#e63946", label="Below normal")
            ax4.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        style_ax(ax4, "Platelet Count", "10³/μL")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=120)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        ai_text = ""
        if GROQ_KEY:
            try:
                client = Groq(api_key=GROQ_KEY)
                msgs = [{"role":"system","content":"You are a hematology expert for SJSU CardioLab. Analyze TGT blood biomarker data. Give thrombogenicity risk: LOW MODERATE or HIGH. Normal: TAT<8, PF1.2<2.0, Hemo<20, Platelets>150."}]
                msgs.append({"role":"user","content":"TGT data from 27mm SJM Regent MHV:"+chr(10)+df.describe().to_string()[:800]})
                resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=350)
                ai_text = chr(10)+"━"*25+chr(10)+"AI ASSESSMENT:"+chr(10)+resp.choices[0].message.content
            except: pass

        return img, "TGT CSV LOADED: "+str(len(df))+" rows"+chr(10)+"Columns: "+", ".join(df.columns.tolist())+ai_text
    except Exception as e:
        return None, "Error: "+str(e)

def analyze_upad_photo(image):
    if image is None: return None, "Upload a uPAD photo first."
    try:
        img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        arr = np.array(img)
        h,w = arr.shape[:2]
        y1,y2,x1,x2 = int(h*0.35),int(h*0.65),int(w*0.35),int(w*0.65)
        zone = arr[y1:y2,x1:x2]
        R,G,B = float(np.mean(zone[:,:,0])),float(np.mean(zone[:,:,1])),float(np.mean(zone[:,:,2]))
        creatinine = max(0, round(0.018*(R-B)-0.3, 2))
        if creatinine < 1.2: stage,action = "Normal","Monitor annually."
        elif creatinine < 1.5: stage,action = "Borderline","Repeat in 3 months."
        elif creatinine < 3.0: stage,action = "Stage 2 CKD","Consult nephrologist."
        elif creatinine < 6.0: stage,action = "Stage 3-4 CKD","Immediate consultation."
        else: stage,action = "Stage 5 CKD","Emergency care needed."
        result_img = img.copy()
        import PIL.ImageDraw as D
        draw = D.Draw(result_img)
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=3)
        return result_img, ("uPAD ANALYSIS"+chr(10)+"━"*22+chr(10)+
            "R:"+str(round(R,1))+" G:"+str(round(G,1))+" B:"+str(round(B,1))+chr(10)+
            "Orange Score: "+str(round(R-B,1))+chr(10)+"━"*22+chr(10)+
            "CREATININE: "+str(creatinine)+" mg/dL"+chr(10)+
            "CKD STAGE: "+stage+chr(10)+"ACTION: "+action+chr(10)+
            "Confirm: Heska Element HT5")
    except Exception as e: return None, "Error: "+str(e)

def generate_image(prompt):
    if not prompt.strip(): return None,"Enter description.","";
    if not HF_TOKEN: return None,"Add HF_TOKEN to Space secrets.","";
    try:
        enhanced,desc = prompt,""
        if GROQ_KEY:
            try:
                client=Groq(api_key=GROQ_KEY)
                resp=client.chat.completions.create(model="llama-3.3-70b-versatile",
                    messages=[{"role":"system","content":"Format: DESCRIPTION: [2 sentences] PROMPT: [detailed image prompt]"},
                    {"role":"user","content":"Biomedical image for CardioLab: "+prompt}],max_tokens=200)
                full=resp.choices[0].message.content
                if "DESCRIPTION:" in full and "PROMPT:" in full:
                    desc=full.split("DESCRIPTION:")[1].split("PROMPT:")[0].strip()
                    enhanced=full.split("PROMPT:")[1].strip()
            except: pass
        headers={"Authorization":"Bearer "+HF_TOKEN,"Content-Type":"application/json"}
        for url in ["https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
                    "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"]:
            try:
                r=requests.post(url,headers=headers,json={"inputs":enhanced,"parameters":{"num_inference_steps":8}},timeout=60)
                if r.status_code==200: return Image.open(io.BytesIO(r.content)),"Generated!",desc
            except: continue
        return None,"Models busy. Try again.",desc
    except Exception as e: return None,"Error: "+str(e),""

def piv_manual(v,s,h):
    vr="HIGH-stenosis" if float(v)>2.0 else "NORMAL"
    sr="HIGH-thrombosis" if float(s)>10 else "ELEVATED" if float(s)>5 else "NORMAL"
    return "Velocity: "+str(v)+" - "+vr+chr(10)+"Shear: "+str(s)+" - "+sr+chr(10)+"HR: "+str(h)+" bpm"

def tgt_manual(t,p,h,pl,tm):
    risk=sum([float(t)>15,float(p)>2.0,float(h)>50,float(pl)<150])
    return "TAT:"+str(t)+" PF1.2:"+str(p)+chr(10)+"Hemo:"+str(h)+" Plt:"+str(pl)+chr(10)+"Time:"+str(tm)+"min"+chr(10)+"RESULT: "+("HIGH RISK" if risk>=3 else "MODERATE" if risk>=2 else "LOW RISK")

with gr.Blocks(title="CardioLab AI", css=CSS) as demo:
    gr.HTML('''<div style="background:linear-gradient(135deg,#1a237e,#b71c1c);padding:20px;text-align:center;border-radius:12px 12px 0 0"><div style="font-size:2.5em;font-weight:900;color:#fff;letter-spacing:3px">CardioLab AI</div></div>''')

    with gr.Tabs():
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="", height=420)
            with gr.Row():
                msg_box = gr.Textbox(placeholder="Ask about CardioLab research...", label="", lines=2, scale=4)
                with gr.Column(scale=1, min_width=80):
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
            send_btn.click(research_chat, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])
            msg_box.submit(research_chat, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

        with gr.Tab("Voice"):
            voice_chatbot = gr.Chatbot(label="", height=320)
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Question")
            with gr.Row():
                voice_btn = gr.Button("Ask by Voice", variant="primary")
                voice_clear = gr.Button("Clear", variant="secondary")
            voice_btn.click(voice_chat, inputs=[audio_input, voice_chatbot], outputs=voice_chatbot)
            voice_clear.click(lambda: [], outputs=voice_chatbot)

        with gr.Tab("Papers"):
            with gr.Row():
                search_input = gr.Textbox(placeholder="e.g. mechanical heart valve thrombogenicity", label="Research Topic", scale=4)
                search_btn = gr.Button("Search", variant="primary", scale=1)
            search_output = gr.Textbox(label="Verified Results", lines=18)
            search_btn.click(quick_search, inputs=search_input, outputs=search_output)
            search_input.submit(quick_search, inputs=search_input, outputs=search_output)

        with gr.Tab("PIV CSV"):
            gr.Markdown("### Upload PIV CSV file — AI generates 4 charts + clinical analysis")
            gr.Markdown("CSV columns: **time, velocity, shear_stress** (any column names work)")
            with gr.Row():
                with gr.Column(scale=1):
                    piv_file = gr.File(label="CLICK HERE TO UPLOAD PIV CSV", file_types=[".csv"])
                    piv_btn = gr.Button("Analyze PIV Data", variant="primary")
                    piv_result = gr.Textbox(label="AI Analysis", lines=10)
                with gr.Column(scale=2):
                    piv_chart = gr.Image(label="PIV Charts", type="pil")
            piv_btn.click(analyze_piv_csv, inputs=piv_file, outputs=[piv_chart, piv_result])

        with gr.Tab("TGT CSV"):
            gr.Markdown("### Upload TGT CSV file — AI generates blood biomarker charts + thrombogenicity assessment")
            gr.Markdown("CSV columns: **time, TAT, PF12, hemoglobin, platelets** (any column names work)")
            with gr.Row():
                with gr.Column(scale=1):
                    tgt_file = gr.File(label="CLICK HERE TO UPLOAD TGT CSV", file_types=[".csv"])
                    tgt_btn = gr.Button("Analyze TGT Data", variant="primary")
                    tgt_result = gr.Textbox(label="AI Assessment", lines=10)
                with gr.Column(scale=2):
                    tgt_chart = gr.Image(label="TGT Blood Charts", type="pil")
            tgt_btn.click(analyze_tgt_csv, inputs=tgt_file, outputs=[tgt_chart, tgt_result])

        with gr.Tab("uPAD Photo"):
            gr.Markdown("### Upload uPAD Photo — Instant CKD diagnosis from Jaffe reaction color")
            with gr.Row():
                with gr.Column():
                    photo_input = gr.Image(label="Upload uPAD Photo", type="numpy", height=280)
                    analyze_btn = gr.Button("Analyze uPAD", variant="primary")
                with gr.Column():
                    photo_img = gr.Image(label="Detection Zone (green box)", type="pil", height=280)
                    photo_text = gr.Textbox(label="CKD Result", lines=10)
            analyze_btn.click(analyze_upad_photo, inputs=photo_input, outputs=[photo_img, photo_text])

        with gr.Tab("AI Image"):
            with gr.Row():
                img_prompt = gr.Textbox(placeholder="e.g. bileaflet heart valve | uPAD device | Arduino TGT circuit", label="Describe image", lines=2, scale=4)
                with gr.Column(scale=1):
                    img_btn = gr.Button("Generate", variant="primary")
                    img_status = gr.Textbox(label="Status", lines=1)
            img_desc = gr.Textbox(label="AI Description", lines=2, interactive=False)
            img_output = gr.Image(label="Generated Image", type="pil", height=380)
            img_btn.click(generate_image, inputs=img_prompt, outputs=[img_output, img_status, img_desc])

        with gr.Tab("PIV Manual"):
            with gr.Row():
                with gr.Column():
                    v=gr.Number(label="Max Velocity m/s", value=1.8)
                    s=gr.Number(label="Wall Shear Stress Pa", value=6.5)
                    h=gr.Number(label="Heart Rate bpm", value=72)
                    piv_out=gr.Textbox(label="Result", lines=4)
                    gr.Button("Analyze", variant="primary").click(piv_manual,inputs=[v,s,h],outputs=piv_out)

        with gr.Tab("TGT Manual"):
            with gr.Row():
                with gr.Column():
                    t1=gr.Number(label="TAT ng/mL", value=18)
                    t2=gr.Number(label="PF1.2", value=2.5)
                    t3=gr.Number(label="Hemoglobin mg/L", value=60)
                    t4=gr.Number(label="Platelets", value=140)
                    t5=gr.Number(label="Time min", value=40)
                    out2=gr.Textbox(label="Result", lines=6)
                    gr.Button("Analyze", variant="primary").click(tgt_manual,inputs=[t1,t2,t3,t4,t5],outputs=out2)

        with gr.Tab("uPAD Manual"):
            with gr.Row():
                with gr.Column():
                    r=gr.Number(label="R value", value=210)
                    g=gr.Number(label="G value", value=140)
                    b=gr.Number(label="B value", value=80)
                    out3=gr.Textbox(label="Result", lines=4)
                    gr.Button("Analyze", variant="primary").click(
                        lambda r,g,b: "Creatinine: "+str(max(0,round(0.02*(r-b)-0.5,2)))+" mg/dL"+chr(10)+("Normal" if max(0,round(0.02*(r-b)-0.5,2))<1.2 else "Borderline" if max(0,round(0.02*(r-b)-0.5,2))<1.5 else "CKD Stage 2+" ),
                        inputs=[r,g,b], outputs=out3)

demo.launch()
