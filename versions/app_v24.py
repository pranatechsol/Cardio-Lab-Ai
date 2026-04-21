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

KNOWHOW = ("SJSU CardioLab: "
    "MCL: Sylgard 184 PDMS 10:1 ratio 48hr cure green laser PIV 70bpm 5L/min. "
    "TGT: Arduino Uno Stepper Motor 150mL blood sampled at 0 20 40 60min measures TAT PF1.2 hemolysis platelets. "
    "uPAD: Jaffe reaction creatinine plus picric acid gives orange-red color normal 0.6-1.2 mg/dL CKD above 1.5. "
    "MHV: 27mm SJM Regent bileaflet also trileaflet monoleaflet pediatric. "
    "Equipment: Heska HT5 hematology analyzer time-resolved PIV Tygon tubing Arduino Uno.")

CSS = """
body, .gradio-container { background: #f0f4f8 !important; }
.tab-nav { background: #ffffff !important; border-bottom: 2px solid #e2e8f0 !important; padding: 0 10px !important; }
.tab-nav button { background: #f7fafc !important; color: #2d3748 !important; border: 1px solid #e2e8f0 !important; border-radius: 8px 8px 0 0 !important; padding: 12px 18px !important; font-weight: 600 !important; margin-top: 6px !important; }
.tab-nav button:hover { background: #ebf4ff !important; color: #1a237e !important; }
.tab-nav button.selected { background: linear-gradient(135deg, #e63946, #c1121f) !important; color: #ffffff !important; font-weight: 700 !important; }
button.primary { background: linear-gradient(135deg, #e63946 0%, #c1121f 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; }
button.secondary { background: #edf2f7 !important; color: #4a5568 !important; border: 1px solid #cbd5e0 !important; border-radius: 8px !important; }
textarea, input[type=number] { background: #f7fafc !important; color: #1a202c !important; border: 1px solid #cbd5e0 !important; border-radius: 8px !important; }
.message.user { background: linear-gradient(135deg, #e63946, #c1121f) !important; color: white !important; }
.message.bot { background: #ebf4ff !important; color: #1a202c !important; border: 1px solid #bee3f8 !important; }
label span { color: #2b6cb0 !important; font-weight: 600 !important; font-size: 0.85em !important; text-transform: uppercase !important; }
"""

# ─── PIV CSV ANALYSIS ────────────────────────────────────────────
def analyze_piv_csv(file):
    if file is None:
        return None, "Please upload a PIV CSV file."
    try:
        df = pd.read_csv(file.name)
        cols = [c.lower().strip() for c in df.columns]
        df.columns = cols

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#0d1b3e")
        fig.suptitle("PIV Data Analysis — SJSU CardioLab MCL", color="white", fontsize=16, fontweight="bold", y=1.02)

        colors = ["#e63946", "#4361ee", "#2ecc71", "#e67e22"]

        # Plot 1 — Velocity over time or position
        ax1 = axes[0, 0]
        ax1.set_facecolor("#1a2744")
        vel_col = next((c for c in cols if "vel" in c or "v_" in c or "u" == c or "speed" in c), cols[0] if len(cols)>0 else None)
        x_col = next((c for c in cols if "time" in c or "x" in c or "pos" in c or "frame" in c), None)
        if vel_col and x_col:
            ax1.plot(df[x_col], df[vel_col], color="#e63946", linewidth=2, label=vel_col)
            ax1.set_xlabel(x_col, color="#a8b2d8")
            ax1.set_ylabel(vel_col, color="#a8b2d8")
        elif vel_col:
            ax1.plot(df[vel_col], color="#e63946", linewidth=2)
            ax1.set_ylabel(vel_col, color="#a8b2d8")
        else:
            ax1.plot(df.iloc[:,0], color="#e63946", linewidth=2)
        ax1.set_title("Velocity Profile", color="white", fontweight="bold")
        ax1.tick_params(colors="#a8b2d8")
        ax1.grid(True, alpha=0.2, color="#2d4a8a")
        ax1.spines["bottom"].set_color("#2d4a8a")
        ax1.spines["left"].set_color("#2d4a8a")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Plot 2 — Shear stress if available
        ax2 = axes[0, 1]
        ax2.set_facecolor("#1a2744")
        shear_col = next((c for c in cols if "shear" in c or "stress" in c or "tau" in c or "wss" in c), None)
        if shear_col:
            ax2.fill_between(range(len(df)), df[shear_col], alpha=0.7, color="#4361ee")
            ax2.plot(df[shear_col], color="#4361ee", linewidth=2)
            ax2.axhline(y=5, color="#e63946", linestyle="--", linewidth=1.5, label="Risk threshold (5 Pa)")
            ax2.axhline(y=10, color="#ff4444", linestyle="--", linewidth=1.5, label="High risk (10 Pa)")
            ax2.set_ylabel("Shear Stress (Pa)", color="#a8b2d8")
            ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        else:
            # Plot second numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                ax2.fill_between(range(len(df)), df[num_cols[1]], alpha=0.7, color="#4361ee")
                ax2.plot(df[num_cols[1]], color="#4361ee", linewidth=2)
                ax2.set_ylabel(num_cols[1], color="#a8b2d8")
        ax2.set_title("Shear Stress / Secondary Variable", color="white", fontweight="bold")
        ax2.tick_params(colors="#a8b2d8")
        ax2.grid(True, alpha=0.2, color="#2d4a8a")
        ax2.spines["bottom"].set_color("#2d4a8a")
        ax2.spines["left"].set_color("#2d4a8a")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # Plot 3 — Distribution histogram
        ax3 = axes[1, 0]
        ax3.set_facecolor("#1a2744")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            ax3.hist(df[num_cols[0]].dropna(), bins=30, color="#2ecc71", alpha=0.8, edgecolor="#1a2744")
            ax3.set_xlabel(num_cols[0], color="#a8b2d8")
            ax3.set_ylabel("Count", color="#a8b2d8")
        ax3.set_title("Value Distribution", color="white", fontweight="bold")
        ax3.tick_params(colors="#a8b2d8")
        ax3.grid(True, alpha=0.2, color="#2d4a8a")
        ax3.spines["bottom"].set_color("#2d4a8a")
        ax3.spines["left"].set_color("#2d4a8a")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # Plot 4 — Summary stats
        ax4 = axes[1, 1]
        ax4.set_facecolor("#1a2744")
        ax4.axis("off")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        summary_text = "SUMMARY STATISTICS" + chr(10) + "━"*22 + chr(10)
        risk_flags = []
        for col in num_cols[:4]:
            mean_val = df[col].mean()
            max_val = df[col].max()
            min_val = df[col].min()
            summary_text += f"{col[:15]}:" + chr(10)
            summary_text += f"  Mean: {mean_val:.3f}" + chr(10)
            summary_text += f"  Max:  {max_val:.3f}" + chr(10)
            summary_text += f"  Min:  {min_val:.3f}" + chr(10)
            if "vel" in col.lower() and max_val > 2.0:
                risk_flags.append("HIGH VELOCITY - stenosis risk")
            if "shear" in col.lower() and max_val > 10:
                risk_flags.append("HIGH SHEAR - thrombosis risk")
        if risk_flags:
            summary_text += chr(10) + "RISK FLAGS:" + chr(10)
            for flag in risk_flags:
                summary_text += "  ⚠ " + flag + chr(10)
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                color="white", fontsize=9, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#0d1b3e", edgecolor="#4361ee", alpha=0.8))

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=120)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        # AI analysis
        ai_summary = ""
        if GROQ_KEY:
            try:
                client = Groq(api_key=GROQ_KEY)
                stats = df.describe().to_string()
                msgs = [{"role":"system","content":"You are a PIV flow analysis expert for SJSU CardioLab. Analyze the statistics from the PIV CSV data and provide a clinical interpretation. Mention velocity ranges, shear stress levels, risk of stenosis or thrombosis, and recommendations."}]
                msgs.append({"role":"user","content":"Analyze this PIV data from our Mock Circulatory Loop with 27mm SJM Regent MHV at 70bpm 5L/min:"+chr(10)+stats[:1000]})
                resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=400)
                ai_summary = chr(10)+"━"*30+chr(10)+"AI CLINICAL INTERPRETATION:"+chr(10)+resp.choices[0].message.content
            except: pass

        result_text = ("PIV CSV ANALYSIS COMPLETE"+chr(10)+
            "Rows: "+str(len(df))+" | Columns: "+str(len(df.columns))+chr(10)+
            "Columns: "+", ".join(df.columns.tolist())+chr(10)+ai_summary)

        return img, result_text

    except Exception as e:
        return None, "Error reading CSV: "+str(e)+chr(10)+"Make sure your CSV has headers and numeric data."

# ─── TGT CSV ANALYSIS ────────────────────────────────────────────
def analyze_tgt_csv(file):
    if file is None:
        return None, "Please upload a TGT CSV file."
    try:
        df = pd.read_csv(file.name)
        cols = [c.lower().strip() for c in df.columns]
        df.columns = cols

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#0d1b3e")
        fig.suptitle("TGT Blood Analysis — SJSU CardioLab", color="white", fontsize=16, fontweight="bold", y=1.02)

        # Expected TGT columns: time, TAT, PF12, hemoglobin, platelets
        time_col = next((c for c in cols if "time" in c or "min" in c), None)
        tat_col = next((c for c in cols if "tat" in c or "thrombin" in c), None)
        pf_col = next((c for c in cols if "pf" in c or "pf1" in c or "prothrombin" in c), None)
        hemo_col = next((c for c in cols if "hemo" in c or "hemoglobin" in c or "hgb" in c), None)
        plt_col = next((c for c in cols if "platelet" in c or "plt" in c), None)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_axis = df[time_col] if time_col else range(len(df))
        x_label = time_col if time_col else "Sample Number"

        normal_limits = {"tat":8, "pf":2.0, "hemo":20, "platelet":150}

        def style_ax(ax, title):
            ax.set_facecolor("#1a2744")
            ax.set_title(title, color="white", fontweight="bold")
            ax.tick_params(colors="#a8b2d8")
            ax.set_xlabel(x_label, color="#a8b2d8")
            ax.grid(True, alpha=0.2, color="#2d4a8a")
            ax.spines["bottom"].set_color("#2d4a8a")
            ax.spines["left"].set_color("#2d4a8a")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Plot 1 — TAT
        ax1 = axes[0, 0]
        col = tat_col if tat_col else (num_cols[0] if num_cols else None)
        if col:
            ax1.plot(x_axis, df[col], color="#e63946", linewidth=2.5, marker="o", markersize=6, label=col)
            ax1.axhline(y=8, color="#ffd700", linestyle="--", linewidth=1.5, label="Normal limit (8 ng/mL)")
            ax1.fill_between(x_axis, df[col], alpha=0.3, color="#e63946")
            ax1.set_ylabel("TAT (ng/mL)", color="#a8b2d8")
            ax1.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        style_ax(ax1, "Thrombin-Antithrombin (TAT)")

        # Plot 2 — PF1.2
        ax2 = axes[0, 1]
        col2 = pf_col if pf_col else (num_cols[1] if len(num_cols)>1 else None)
        if col2:
            ax2.plot(x_axis, df[col2], color="#4361ee", linewidth=2.5, marker="s", markersize=6, label=col2)
            ax2.axhline(y=2.0, color="#ffd700", linestyle="--", linewidth=1.5, label="Normal limit (2.0)")
            ax2.fill_between(x_axis, df[col2], alpha=0.3, color="#4361ee")
            ax2.set_ylabel("PF1.2 (nmol/L)", color="#a8b2d8")
            ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        style_ax(ax2, "Prothrombin Fragment (PF1.2)")

        # Plot 3 — Hemoglobin / Hemolysis
        ax3 = axes[1, 0]
        col3 = hemo_col if hemo_col else (num_cols[2] if len(num_cols)>2 else None)
        if col3:
            ax3.bar(range(len(df)), df[col3], color="#2ecc71", alpha=0.8, edgecolor="#1a2744", label=col3)
            ax3.axhline(y=20, color="#ffd700", linestyle="--", linewidth=1.5, label="Normal limit (20 mg/L)")
            ax3.set_ylabel("Free Hemoglobin (mg/L)", color="#a8b2d8")
            ax3.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        style_ax(ax3, "Free Hemoglobin (Hemolysis)")

        # Plot 4 — Platelets
        ax4 = axes[1, 1]
        col4 = plt_col if plt_col else (num_cols[3] if len(num_cols)>3 else None)
        if col4:
            ax4.plot(x_axis, df[col4], color="#e67e22", linewidth=2.5, marker="^", markersize=6, label=col4)
            ax4.axhline(y=150, color="#ffd700", linestyle="--", linewidth=1.5, label="Normal minimum (150)")
            ax4.fill_between(x_axis, df[col4], 150, where=df[col4]<150, alpha=0.3, color="#e63946", label="Below normal")
            ax4.set_ylabel("Platelet Count (10³/μL)", color="#a8b2d8")
            ax4.legend(fontsize=8, labelcolor="white", facecolor="#1a2744")
        style_ax(ax4, "Platelet Count")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=120)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        # AI analysis
        ai_summary = ""
        if GROQ_KEY:
            try:
                client = Groq(api_key=GROQ_KEY)
                stats = df.describe().to_string()
                msgs = [{"role":"system","content":"You are a hematology expert for SJSU CardioLab TGT testing. Analyze the blood biomarker data and provide thrombogenicity assessment. Comment on TAT levels, PF1.2, hemolysis, platelet consumption. Give overall thrombogenic risk: LOW MODERATE or HIGH. Reference normal ranges: TAT below 8 ng/mL, PF1.2 below 2.0 nmol/L, Hemoglobin below 20 mg/L, Platelets above 150."}]
                msgs.append({"role":"user","content":"Analyze TGT blood data from 27mm SJM Regent MHV tested in our MCL at 70bpm 5L/min:"+chr(10)+stats[:1000]})
                resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=400)
                ai_summary = chr(10)+"━"*30+chr(10)+"AI THROMBOGENICITY ASSESSMENT:"+chr(10)+resp.choices[0].message.content
            except: pass

        result_text = ("TGT CSV ANALYSIS COMPLETE"+chr(10)+
            "Rows: "+str(len(df))+" | Columns: "+str(len(df.columns))+chr(10)+
            "Columns detected: "+", ".join(df.columns.tolist())+chr(10)+ai_summary)

        return img, result_text

    except Exception as e:
        return None, "Error reading CSV: "+str(e)+chr(10)+"Make sure your CSV has headers like: time, TAT, PF12, hemoglobin, platelets"

# ─── OTHER FUNCTIONS ──────────────────────────────────────────────
def get_pubmed(query, n=5):
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":query+" AND (mechanical heart valve OR microfluidic OR CKD OR thrombogenicity)","retmax":n,"retmode":"json","sort":"date"},timeout=10)
        ids = r.json()["esearchresult"]["idlist"]
        if not ids: return ""
        return chr(10).join(["https://pubmed.ncbi.nlm.nih.gov/"+i for i in ids])
    except: return ""

def get_scholar(query, n=5):
    try:
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query":query+" biomedical","limit":n,"fields":"title,year,url,citationCount"},timeout=10)
        papers = r.json().get("data",[])
        out = []
        for p in papers:
            url = p.get("url","")
            if url: out.append(p.get("title","")[:80]+" ("+str(p.get("year",""))+") - "+str(p.get("citationCount",0))+" citations"+chr(10)+"   "+url)
        return chr(10).join(out)
    except: return ""

def quick_search(query):
    if not query.strip(): return "Please enter a research topic."
    pubmed = get_pubmed(query, n=8)
    scholar = get_scholar(query, n=5)
    return "PUBMED RESULTS:"+chr(10)+pubmed+chr(10)+chr(10)+"SEMANTIC SCHOLAR:"+chr(10)+scholar

def research_chat(message, history):
    if not GROQ_KEY:
        history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":"Error: Add GROQ_API_KEY to Space Settings Secrets."})
        return "", history
    try:
        client = Groq(api_key=GROQ_KEY)
        pubmed = get_pubmed(message, n=3)
        msgs = [{"role":"system","content":"You are CardioLab AI. Expert in MHV MCL PIV TGT uPAD CKD FSI. Remember full conversation. Never invent URLs. "+KNOWHOW}]
        for item in history:
            if isinstance(item, dict): msgs.append({"role":item["role"],"content":item["content"]})
        msgs.append({"role":"user","content":message})
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=700)
        answer = resp.choices[0].message.content
        if pubmed: answer += chr(10)+chr(10)+"PUBMED LINKS:"+chr(10)+pubmed
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
        text = tx.text
        msgs = [{"role":"system","content":"You are CardioLab AI. "+KNOWHOW}]
        for item in history:
            if isinstance(item, dict): msgs.append({"role":item["role"],"content":item["content"]})
        msgs.append({"role":"user","content":text})
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=500)
        history.append({"role":"user","content":"[Voice] "+text})
        history.append({"role":"assistant","content":resp.choices[0].message.content})
        return history
    except Exception as e:
        history.append({"role":"assistant","content":"Voice error: "+str(e)})
        return history

def analyze_upad_photo(image):
    if image is None:
        return None, "Please upload a uPAD photo first."
    try:
        img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        y1,y2 = int(h*0.35),int(h*0.65)
        x1,x2 = int(w*0.35),int(w*0.65)
        zone = img_array[y1:y2, x1:x2]
        R = float(np.mean(zone[:,:,0]))
        G = float(np.mean(zone[:,:,1]))
        B = float(np.mean(zone[:,:,2]))
        orange_score = R - B
        creatinine = max(0, round(0.018 * orange_score - 0.3, 2))
        if creatinine < 1.2: stage,action = "Normal","No CKD. Monitor annually."
        elif creatinine < 1.5: stage,action = "Borderline","Repeat in 3 months. Consult physician."
        elif creatinine < 3.0: stage,action = "Stage 2 CKD","Consult nephrologist. Confirm with Heska HT5."
        elif creatinine < 6.0: stage,action = "Stage 3-4 CKD","Advanced CKD. Immediate medical consultation."
        else: stage,action = "Stage 5 CKD","Kidney failure range. Emergency care needed."
        result_img = img.copy()
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(result_img)
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=3)
        result = ("uPAD PHOTO ANALYSIS"+chr(10)+"━"*27+chr(10)+
            "R: "+str(round(R,1))+" G: "+str(round(G,1))+" B: "+str(round(B,1))+chr(10)+
            "Orange Score: "+str(round(orange_score,1))+chr(10)+"━"*27+chr(10)+
            "CREATININE: "+str(creatinine)+" mg/dL"+chr(10)+
            "CKD STAGE: "+stage+chr(10)+"━"*27+chr(10)+
            "ACTION: "+action+chr(10)+"Confirm with: Heska Element HT5")
        return result_img, result
    except Exception as e:
        return None, "Error: "+str(e)

def piv_tool(velocity, shear, hr):
    v = "HIGH - stenosis" if float(velocity)>2.0 else "NORMAL"
    s = "HIGH - thrombosis" if float(shear)>10 else "ELEVATED" if float(shear)>5 else "NORMAL"
    return "PIV: Velocity "+str(velocity)+" m/s - "+v+chr(10)+"Shear "+str(shear)+" Pa - "+s+chr(10)+"HR "+str(hr)+" bpm"

def tgt_tool(tat,pf12,hemo,platelets,time):
    risk=sum([float(tat)>15,float(pf12)>2.0,float(hemo)>50,float(platelets)<150])
    r="HIGH RISK" if risk>=3 else "MODERATE" if risk>=2 else "LOW RISK"
    return "TGT: TAT "+str(tat)+" PF1.2 "+str(pf12)+chr(10)+"Hemo "+str(hemo)+" Plt "+str(platelets)+chr(10)+"Time "+str(time)+" min"+chr(10)+"RESULT: "+r

def generate_image(prompt):
    if not prompt.strip(): return None,"Enter description.","";
    if not HF_TOKEN: return None,"Error: Add HF_TOKEN to Space secrets.","";
    try:
        enhanced=prompt
        description=""
        if GROQ_KEY:
            try:
                client=Groq(api_key=GROQ_KEY)
                msgs=[{"role":"system","content":"Biomedical visualization expert. Format: DESCRIPTION: [desc] PROMPT: [prompt]"},{"role":"user","content":"Create image for: "+prompt}]
                resp=client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=200)
                full=resp.choices[0].message.content
                if "DESCRIPTION:" in full and "PROMPT:" in full:
                    description=full.split("DESCRIPTION:")[1].split("PROMPT:")[0].strip()
                    enhanced=full.split("PROMPT:")[1].strip()
            except: pass
        headers={"Authorization":"Bearer "+HF_TOKEN,"Content-Type":"application/json"}
        payload={"inputs":enhanced,"parameters":{"num_inference_steps":8}}
        for url in ["https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell","https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"]:
            try:
                r=requests.post(url,headers=headers,json=payload,timeout=60)
                if r.status_code==200:
                    return Image.open(io.BytesIO(r.content)),"Generated!",description
            except: continue
        return None,"Models busy. Try again.",description
    except Exception as e: return None,"Error: "+str(e),""

# ─── UI ───────────────────────────────────────────────────────────
with gr.Blocks(title="CardioLab AI", css=CSS) as demo:
    gr.HTML('''<div style="background:linear-gradient(135deg,#1a237e,#b71c1c);padding:25px;text-align:center;border-radius:12px 12px 0 0"><div style="font-size:2.8em;font-weight:900;color:#fff;letter-spacing:3px">CardioLab AI</div></div>''')

    with gr.Tabs():

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="", height=450)
            with gr.Row():
                msg_box = gr.Textbox(placeholder="Ask anything about CardioLab research...", label="", lines=2, scale=4)
                with gr.Column(scale=1, min_width=100):
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
            send_btn.click(research_chat, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])
            msg_box.submit(research_chat, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

        with gr.Tab("Voice"):
            gr.Markdown("### Speak your question - Groq Whisper AI")
            voice_chatbot = gr.Chatbot(label="", height=350)
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

        with gr.Tab("PIV Data"):
            gr.Markdown("### Upload PIV CSV — AI generates charts and clinical interpretation")
            gr.Markdown("**Expected columns:** time/x, velocity, shear_stress (any names work — AI detects automatically)")
            with gr.Row():
                with gr.Column(scale=1):
                    piv_file = gr.File(label="Upload PIV CSV File", file_types=[".csv"])
                    piv_analyze_btn = gr.Button("Analyze PIV Data", variant="primary")
                    gr.Markdown("**Sample CSV format:**")
                    gr.Markdown("```\ntime,velocity,shear_stress\n0,0.5,2.1\n1,1.2,4.5\n2,1.8,7.2\n```")
                with gr.Column(scale=2):
                    piv_chart = gr.Image(label="PIV Charts", type="pil", height=450)
                    piv_ai_result = gr.Textbox(label="AI Clinical Analysis", lines=10)
            piv_analyze_btn.click(analyze_piv_csv, inputs=piv_file, outputs=[piv_chart, piv_ai_result])

        with gr.Tab("TGT Data"):
            gr.Markdown("### Upload TGT CSV — AI generates blood biomarker charts and thrombogenicity assessment")
            gr.Markdown("**Expected columns:** time, TAT, PF12, hemoglobin, platelets (any names work — AI detects automatically)")
            with gr.Row():
                with gr.Column(scale=1):
                    tgt_file = gr.File(label="Upload TGT CSV File", file_types=[".csv"])
                    tgt_analyze_btn = gr.Button("Analyze TGT Data", variant="primary")
                    gr.Markdown("**Sample CSV format:**")
                    gr.Markdown("```\ntime,TAT,PF12,hemoglobin,platelets\n0,5.2,1.1,12,210\n20,9.8,1.8,18,195\n40,14.2,2.4,35,178\n60,18.6,3.1,62,145\n```")
                with gr.Column(scale=2):
                    tgt_chart = gr.Image(label="TGT Blood Analysis Charts", type="pil", height=450)
                    tgt_ai_result = gr.Textbox(label="AI Thrombogenicity Assessment", lines=10)
            tgt_analyze_btn.click(analyze_tgt_csv, inputs=tgt_file, outputs=[tgt_chart, tgt_ai_result])

        with gr.Tab("uPAD Photo"):
            gr.Markdown("### Upload uPAD Photo — Instant CKD diagnosis from Jaffe reaction color")
            with gr.Row():
                with gr.Column(scale=1):
                    photo_input = gr.Image(label="Upload uPAD Photo", type="numpy", height=300)
                    analyze_btn = gr.Button("Analyze uPAD Photo", variant="primary")
                with gr.Column(scale=1):
                    photo_result_img = gr.Image(label="Analyzed Image", type="pil", height=300)
                    photo_result_text = gr.Textbox(label="CKD Result", lines=12)
            analyze_btn.click(analyze_upad_photo, inputs=photo_input, outputs=[photo_result_img, photo_result_text])

        with gr.Tab("uPAD Manual"):
            with gr.Row():
                with gr.Column():
                    r=gr.Number(label="R value", value=210)
                    g=gr.Number(label="G value", value=140)
                    b=gr.Number(label="B value", value=80)
                    out3=gr.Textbox(label="Result", lines=6)
                    gr.Button("Analyze", variant="primary").click(
                        lambda r,g,b: "Creatinine: "+str(max(0,round(0.02*(r-b)-0.5,2)))+" mg/dL"+chr(10)+("Normal" if max(0,round(0.02*(r-b)-0.5,2))<1.2 else "Borderline" if max(0,round(0.02*(r-b)-0.5,2))<1.5 else "CKD"),
                        inputs=[r,g,b], outputs=out3)

        with gr.Tab("AI Image"):
            with gr.Row():
                img_prompt = gr.Textbox(placeholder="e.g. bileaflet heart valve | uPAD microfluidic | Arduino TGT circuit", label="Describe image", lines=3, scale=4)
                with gr.Column(scale=1):
                    img_btn = gr.Button("Generate", variant="primary")
                    img_status = gr.Textbox(label="Status", lines=2)
            img_desc = gr.Textbox(label="AI Description", lines=2, interactive=False)
            img_output = gr.Image(label="Generated Image", type="pil", height=400)
            img_btn.click(generate_image, inputs=img_prompt, outputs=[img_output, img_status, img_desc])

        with gr.Tab("PIV Manual"):
            with gr.Row():
                with gr.Column():
                    v=gr.Number(label="Max Velocity m/s", value=1.8)
                    s=gr.Number(label="Wall Shear Stress Pa", value=6.5)
                    h=gr.Number(label="Heart Rate bpm", value=72)
                    piv_out=gr.Textbox(label="Result", lines=5)
                    gr.Button("Analyze PIV", variant="primary").click(piv_tool,inputs=[v,s,h],outputs=piv_out)

        with gr.Tab("TGT Manual"):
            with gr.Row():
                with gr.Column():
                    t1=gr.Number(label="TAT ng/mL", value=18)
                    t2=gr.Number(label="PF1.2 nmol/L", value=2.5)
                    t3=gr.Number(label="Free Hemoglobin mg/L", value=60)
                    t4=gr.Number(label="Platelet Count", value=140)
                    t5=gr.Number(label="Time minutes", value=40)
                    out2=gr.Textbox(label="Result", lines=8)
                    gr.Button("Analyze TGT", variant="primary").click(tgt_tool,inputs=[t1,t2,t3,t4,t5],outputs=out2)

demo.launch()
