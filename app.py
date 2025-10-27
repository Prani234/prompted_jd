import os
import streamlit as st
import PyPDF2
import docx
import json
import requests
from dotenv import load_dotenv
from TTS.api import TTS

# ---------------- Load environment variables ----------------
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"

UPLOAD_PATH = "__DATA__"
os.makedirs(UPLOAD_PATH, exist_ok=True)

# ---------------- Text Extraction ----------------
def extract_text(file_path, file_type="pdf"):
    """Extract text from PDF, DOCX, or TXT."""
    text = ""
    try:
        if file_type == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
        elif file_type == "docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        return text.strip()
    except Exception as e:
        return f"__READ_ERROR__:{str(e)}"


# ---------------- Job Info Extractor (Groq LLM) ----------------
def extract_job_fields(job_text):
    """
    Use Groq LLM to extract structured job information
    matching the exact fields specified.
    """
    prompt = f"""
You are an expert job information extraction assistant.

From the job description below, extract and return a structured JSON object
with the following fields:

- job_title
- company_name
- location
- employment_type
- seniority_level
- hard_skills
- soft_skills
- certifications
- tools
- experience
- education_level
- duties_and_responsibilities
- preferred_skills
- communication_skills
- language
- salary

Rules:
- Use lists for fields like skills, certifications, tools, or languages.
- If a field is not present, return null or an empty list.
- Maintain JSON validity strictly — no explanations or extra text.
- Do not infer or guess beyond the text; extract only what is clearly mentioned.

Job Description:
\"\"\"{job_text}\"\"\"
"""

    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1200,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Attempt to recover JSON from text
            import re
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"error": "No valid JSON detected", "raw_output": content}
    except Exception as e:
        return {"error": str(e)}


# ---------------- Job Intro Script ----------------
def generate_job_intro(job_json):
    """
    Generate a short, natural-sounding job introduction based on extracted data.
    """
    prompt = (
        "You are an HR assistant creating a professional spoken job introduction "
        "for a voice-over. Write a 2-paragraph overview (around 100–150 words) "
        "highlighting the role, key skills, company, and what makes this position appealing. "
        "Use a friendly yet professional tone.\n\n"
        f"Job Data:\n{json.dumps(job_json, indent=2)}"
    )

    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 400,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating intro: {str(e)}"


# ---------------- Streamlit UI ----------------
st.title("📋 Job Description Field Extractor + Audio Intro (Groq LLM)")

uploaded_file = st.file_uploader("Upload a Job Description file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    file_path = os.path.join(UPLOAD_PATH, f"jobdesc.{file_ext}")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    job_text = extract_text(file_path, file_type=file_ext)

    if job_text.startswith("__READ_ERROR__"):
        st.error(f"Failed to read file: {job_text}")
    elif len(job_text.strip()) < 50:
        st.warning("⚠️ The job description text seems too short.")
    else:
        st.text_area("📝 Extracted Text", job_text, height=200)

        if st.button("🔍 Extract Job Fields"):
            job_json = extract_job_fields(job_text)
            st.subheader("📊 Extracted Job Information (JSON)")
            st.json(job_json)

            if "error" not in job_json:
                st.subheader("🎙 Generate Job Introduction Script")
                job_intro = generate_job_intro(job_json)
                st.text_area("Generated Introduction Script", job_intro, height=150)

                if st.button("🔊 Convert to Audio"):
                    try:
                        audio_path = os.path.join(UPLOAD_PATH, "job_intro.wav")
                        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
                        tts.tts_to_file(text=job_intro, file_path=audio_path)
                        st.audio(audio_path)
                    except Exception as e:
                        st.error(f"TTS conversion failed: {str(e)}")
