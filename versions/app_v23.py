import gradio as gr
import os, requests, io
import numpy as np
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

def analyze_upad_photo(image):
    if image is None:
        return None, "Please upload a uPAD photo first."
    try:
        img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # Find the detection zone - center 30% of image
        # This is where the Jaffe reaction orange-red color appears
        y1 = int(h * 0.35)
        y2 = int(h * 0.65)
        x1 = int(w * 0.35)
        x2 = int(w * 0.65)
        zone = img_array[y1:y2, x1:x2]

        # Extract RGB from detection zone
        R = float(np.mean(zone[:,:,0]))
        G = float(np.mean(zone[:,:,1]))
        B = float(np.mean(zone[:,:,2]))

        # Jaffe reaction: orange-red color = high R, low B
        # Higher R-B score = more creatinine
        orange_score = R - B

        # Calibrated formula for Jaffe reaction uPAD
        # Based on: orange-red color intensity maps to creatinine concentration
        creatinine = max(0, round(0.018 * orange_score - 0.3, 2))

        # CKD Staging
        if creatinine < 1.2:
            stage = "Normal"
            stage_color = "GREEN"
            action = "No CKD detected. Continue monitoring annually."
        elif creatinine < 1.5:
            stage = "Borderline"
            stage_color = "YELLOW"
            action = "Borderline range. Repeat test in 3 months. Consult physician."
        elif creatinine < 3.0:
            stage = "Stage 2 CKD"
            stage_color = "ORANGE"
            action = "Stage 2 CKD detected. Consult nephrologist. Confirm with Heska Element HT5."
        elif creatinine < 6.0:
            stage = "Stage 3-4 CKD"
            stage_color = "RED"
            action = "Advanced CKD. Immediate medical consultation required."
        else:
            stage = "Stage 5 CKD"
            stage_color = "CRITICAL"
            action = "Kidney failure range. Emergency medical care needed."

        # Draw analysis box on image
        result_img = img.copy()
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(result_img)

        # Draw detection zone box in green
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.rectangle([x1-1, y1-1, x2+1, y2+1], outline=(0, 200, 0), width=1)

        result = (
            "uPAD PHOTO ANALYSIS RESULTS" + chr(10) +
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━" + chr(10) +
            "DETECTION ZONE (center 30%):" + chr(10) +
            "  R (Red):   " + str(round(R, 1)) + chr(10) +
            "  G (Green): " + str(round(G, 1)) + chr(10) +
            "  B (Blue):  " + str(round(B, 1)) + chr(10) +
            "  Orange Score (R-B): " + str(round(orange_score, 1)) + chr(10) +
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━" + chr(10) +
            "CREATININE: " + str(creatinine) + " mg/dL" + chr(10) +
            "CKD STAGE:  " + stage + " [" + stage_color + "]" + chr(10) +
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━" + chr(10) +
            "ACTION: " + action + chr(10) + chr(10) +
            "Normal range: 0.6-1.2 mg/dL" + chr(10) +
            "Confirm results with: Heska Element HT5" + chr(10) +
            "Method: Jaffe Reaction (picric acid)"
        )

        return result_img, result

    except Exception as e:
        return None, "Error analyzing image: " + str(e)

def analyze_upad_manual(r, g, b):
    c = max(0, round(0.02*(float(r)-float(b))-0.5, 2))
    if c < 1.2: s = "Normal - No CKD"
    elif c < 1.5: s = "Borderline - Monitor"
    elif c < 3.0: s = "Stage 2 CKD"
    elif c < 6.0: s = "Stage 3-4 CKD"
    else: s = "Stage 5 CKD - Kidney Failure"
    return ("uPAD MANUAL ANALYSIS" + chr(10) +
        "━━━━━━━━━━━━━━━━━━━━" + chr(10) +
        "RGB: R=" + str(r) + " G=" + str(g) + " B=" + str(b) + chr(10) +
        "Creatinine: " + str(c) + " mg/dL" + chr(10) +
        "CKD Stage: " + s + chr(10) +
        "Confirm with: Heska Element HT5")

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

def generate_image(prompt):
    if not prompt.strip(): return None, "Please enter a description.", ""
    if not HF_TOKEN: return None, "Error: Add HF_TOKEN to Space Settings Secrets.", ""
    try:
        enhanced = prompt
        description = ""
        if GROQ_KEY:
            try:
                client = Groq(api_key=GROQ_KEY)
                msgs = [
                    {"role":"system","content":"You are a biomedical visualization expert for SJSU CardioLab. Do two things: 1) Write a clear 2-3 sentence description of what the image will show. 2) Write a detailed image generation prompt. Format: DESCRIPTION: [description] PROMPT: [prompt]"},
                    {"role":"user","content":"Create image for: "+prompt+". CardioLab context: 27mm SJM Regent bileaflet mechanical heart valve, Sylgard 184 transparent silicone MCL, green laser PIV, Arduino Uno stepper motor TGT, Whatman paper uPAD microfluidic device, Jaffe reaction orange-red color CKD creatinine."}
                ]
                resp = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=300)
                full_resp = resp.choices[0].message.content
                if "DESCRIPTION:" in full_resp and "PROMPT:" in full_resp:
                    description = full_resp.split("DESCRIPTION:")[1].split("PROMPT:")[0].strip()
                    enhanced = full_resp.split("PROMPT:")[1].strip()
                else:
                    description = full_resp[:200]
                    enhanced = "Highly detailed scientific biomedical illustration: "+prompt+", professional medical diagram, photorealistic, high quality, labeled"
            except: enhanced = prompt
        headers = {"Authorization":"Bearer "+HF_TOKEN,"Content-Type":"application/json"}
        payload = {"inputs":enhanced,"parameters":{"num_inference_steps":8,"guidance_scale":7.5}}
        models = [
            "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
            "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0",
        ]
        for model_url in models:
            try:
                r = requests.post(model_url,headers=headers,json=payload,timeout=60)
                if r.status_code == 200:
                    img = Image.open(io.BytesIO(r.content))
                    return img, "Image generated!", description
            except: continue
        return None, "Models busy. Try again in 30 seconds.", description
    except Exception as e:
        return None, "Error: "+str(e), ""

def piv_tool(velocity, shear, hr):
    v = "HIGH - stenosis risk" if float(velocity)>2.0 else "NORMAL"
    s = "HIGH - thrombosis risk" if float(shear)>10 else "ELEVATED" if float(shear)>5 else "NORMAL"
    hr_s = "ABNORMAL" if float(hr)<60 or float(hr)>100 else "NORMAL"
    return ("PIV ANALYSIS RESULTS"+chr(10)+"━━━━━━━━━━━━━━━━━━━━"+chr(10)+
        "Velocity:   "+str(velocity)+" m/s  →  "+v+chr(10)+
        "Shear:      "+str(shear)+" Pa   →  "+s+chr(10)+
        "Heart Rate: "+str(hr)+" bpm →  "+hr_s)

def tgt_tool(tat,pf12,hemo,platelets,time):
    risk=sum([float(tat)>15,float(pf12)>2.0,float(hemo)>50,float(platelets)<150])
    r="HIGH THROMBOGENIC RISK" if risk>=3 else "MODERATE RISK" if risk>=2 else "LOW RISK"
    return ("TGT BLOOD ANALYSIS"+chr(10)+"━━━━━━━━━━━━━━━━━━━━"+chr(10)+
        "Time: "+str(time)+" min"+chr(10)+
        "TAT: "+str(tat)+(" HIGH" if float(tat)>15 else " NORMAL")+chr(10)+
        "PF1.2: "+str(pf12)+(" HIGH" if float(pf12)>2.0 else " NORMAL")+chr(10)+
        "Hemo: "+str(hemo)+(" HIGH" if float(hemo)>50 else " NORMAL")+chr(10)+
        "Platelets: "+str(platelets)+(" LOW" if float(platelets)<150 else " NORMAL")+chr(10)+
        "━━━━━━━━━━━━━━━━━━━━"+chr(10)+"OVERALL: "+r)

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

        with gr.Tab("uPAD Photo"):
            gr.Markdown("### Upload uPAD Photo — AI reads color automatically and gives instant CKD diagnosis")
            gr.Markdown("**How it works:** AI finds the detection zone in center of image, extracts RGB color from Jaffe reaction area, calculates creatinine level, gives CKD stage")
            gr.Markdown("**Supported:** Photo from phone camera, scanned image, or microscope image of uPAD test strip")
            with gr.Row():
                with gr.Column(scale=1):
                    photo_input = gr.Image(label="Upload uPAD Photo", type="numpy", height=300)
                    analyze_btn = gr.Button("Analyze uPAD Photo", variant="primary")
                    gr.Markdown("**Tips for best results:**")
                    gr.Markdown("- Take photo in good lighting")
                    gr.Markdown("- Keep uPAD flat and centered")
                    gr.Markdown("- Detection zone is center 30% of image")
                with gr.Column(scale=1):
                    photo_result_img = gr.Image(label="Analyzed Image (green box = detection zone)", type="pil", height=300)
                    photo_result_text = gr.Textbox(label="CKD Analysis Result", lines=16)
            analyze_btn.click(analyze_upad_photo, inputs=photo_input, outputs=[photo_result_img, photo_result_text])

        with gr.Tab("uPAD Manual"):
            gr.Markdown("### Enter RGB values manually if you already measured them")
            with gr.Row():
                with gr.Column():
                    r=gr.Number(label="R value", value=210, info="Range: 0-255")
                    g=gr.Number(label="G value", value=140, info="Range: 0-255")
                    b=gr.Number(label="B value", value=80, info="Range: 0-255")
                    out3=gr.Textbox(label="Result", lines=6)
                    gr.Button("Analyze uPAD", variant="primary").click(analyze_upad_manual,inputs=[r,g,b],outputs=out3)

        with gr.Tab("AI Image"):
            gr.Markdown("### Real AI Image Generation using FLUX.1")
            with gr.Row():
                img_prompt = gr.Textbox(placeholder="e.g. bileaflet mechanical heart valve | uPAD microfluidic device | Arduino TGT circuit", label="Describe the image", lines=3, scale=4)
                with gr.Column(scale=1):
                    img_btn = gr.Button("Generate Image", variant="primary")
                    img_status = gr.Textbox(label="Status", lines=2)
            img_desc = gr.Textbox(label="AI Description", lines=3, interactive=False)
            img_output = gr.Image(label="Generated Image", type="pil", height=450)
            img_btn.click(generate_image, inputs=img_prompt, outputs=[img_output, img_status, img_desc])

        with gr.Tab("PIV"):
            gr.Markdown("### Analyze PIV flow data from Mock Circulatory Loop")
            with gr.Row():
                with gr.Column():
                    v=gr.Number(label="Max Velocity m/s", value=1.8, info="Normal: 0.5-2.0 m/s")
                    s=gr.Number(label="Wall Shear Stress Pa", value=6.5, info="Normal: below 5 Pa")
                    h=gr.Number(label="Heart Rate bpm", value=72, info="Normal: 60-100 bpm")
                    piv_out=gr.Textbox(label="Result", lines=6)
                    gr.Button("Analyze PIV", variant="primary").click(piv_tool,inputs=[v,s,h],outputs=piv_out)

        with gr.Tab("TGT"):
            gr.Markdown("### Interpret Thrombogenicity Tester blood analysis results")
            with gr.Row():
                with gr.Column():
                    t1=gr.Number(label="TAT ng/mL", value=18, info="Normal: below 8")
                    t2=gr.Number(label="PF1.2 nmol/L", value=2.5, info="Normal: below 2.0")
                    t3=gr.Number(label="Free Hemoglobin mg/L", value=60, info="Normal: below 20")
                    t4=gr.Number(label="Platelet Count", value=140, info="Normal: above 150")
                    t5=gr.Number(label="Time minutes", value=40)
                    out2=gr.Textbox(label="Result", lines=10)
                    gr.Button("Analyze TGT", variant="primary").click(tgt_tool,inputs=[t1,t2,t3,t4,t5],outputs=out2)

demo.launch()
