# CardioLab AI Agent
## SJSU Biomedical Engineering

Live Demo: https://huggingface.co/spaces/Saicharan21/CardioLab-AI

## What is this?
CardioLab AI is an open source AI research agent built specifically for
SJSU CardioLab Biomedical Engineering. Built on Biomni from Stanford SNAP Lab.

## Features
- Chat with memory like ChatGPT
- PIV Analysis tool — interprets Mock Circulatory Loop flow data
- TGT Results tool — interprets Thrombogenicity Tester blood results
- uPAD CKD tool — analyzes colorimetric creatinine test results
- PubMed live search — finds real verified research papers

## Research Pillars
- Pillar 1: CKD Diagnostics (uPADs, Jaffe reaction, creatinine)
- Pillar 2: Mechanical Heart Valves (MCL, PIV, TGT)
- Pillar 3: FSI Simulations (COMSOL, bileaflet/trileaflet MHV)

## Equipment
- 27mm SJM Regent MHV
- Thrombogenicity Tester with Arduino Uno + Stepper Motor
- Time-resolved PIV system with green laser
- Mock Circulatory Loop with Sylgard 184
- Heska Element HT5 hematology analyzer

## Tech Stack
- Brain: Llama 3.3 70B via Groq (free)
- Base: Biomni from Stanford SNAP Lab (Apache 2.0)
- Vector DB: ChromaDB with PubMedBERT embeddings
- UI: Gradio
- Host: Hugging Face Spaces (free)

## How to run locally
1. Clone this repo
2. pip install -r requirements.txt
3. Set GROQ_API_KEY environment variable
4. python app.py

## License
Apache 2.0 - free to use, modify, share
