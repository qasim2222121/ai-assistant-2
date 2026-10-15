# ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an ATS score, a category breakdown,
missing keywords, prioritized improvements and rewrite examples. Built with
Streamlit and Google Gemini Flash.

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```
Get a free key at https://aistudio.google.com/apikey

## Configuration
| Name | Purpose | Default |
|------|---------|---------|
| `GEMINI_API_KEY` | Gemini API key (env var or Streamlit secret) | none |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.8-flash` |

For local secrets, create `.streamlit/secrets.toml` (never commit it):
```toml
GEMINI_API_KEY = "your-key"
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and click **Create app**.
3. Pick the repo, branch `main`, main file `app.py`.
4. Under **Advanced settings > Secrets** paste: `GEMINI_API_KEY = "your-key"`
5. Click **Deploy**.

## Notes
- Scanned/image-only PDFs have no text layer and can't be analyzed; use a text PDF or DOCX.
- The ATS score is an AI estimate, not the output of a real ATS.
