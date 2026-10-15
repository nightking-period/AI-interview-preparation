# AI-Based Interview Preparation System — Web Deployment

This is a Streamlit web version of the existing Tkinter project. The question bank is preserved, and the web interface reuses the original search, CSP, keyword-evaluation, and logical-reasoning ideas.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install the dependency:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   python -m streamlit run streamlit_app.py
   ```
5. Streamlit opens the app in your browser, usually at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository (for example, `ai-interview-preparation-system`).
2. Upload `streamlit_app.py`, `questions.json`, `requirements.txt`, and this README.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `streamlit_app.py`.
5. Deploy. After the build completes, copy the app URL to share with your professor.

## Notes

- `AI_Interview_System.py` is the original desktop/Tkinter app; this folder contains the separate web version.
- Answer scoring is still keyword-based and rule-based, not a large-language-model semantic evaluation.
- The app uses Streamlit as its only third-party dependency.
