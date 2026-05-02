# AI Customer Support Copilot

A Streamlit-based AI customer support copilot for e-commerce teams. The app uses the Gemini API to classify customer queries, estimate priority, suggest internal workflows, draft professional replies, and show basic analytics.

## Features

- Gemini API response generation
- Structured JSON parsing and validation
- Priority color coding
- Sample query buttons
- Chat-style output
- CSV logging
- Analytics dashboard

## Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `.gitignore` - Keeps local secrets, logs, and cache files out of GitHub

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Gemini API key.

PowerShell:

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

Run the app:

```bash
streamlit run app.py
```

## Deploy On Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to `https://share.streamlit.io`.
3. Sign in with GitHub.
4. Click `Create app`.
5. Choose your repository, branch, and `app.py` as the main file.
6. Open `Advanced settings`.
7. Add this secret:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

8. Click `Deploy`.

After deployment, Streamlit will create a public website link ending in `.streamlit.app`.

## Important

Never commit your Gemini API key to GitHub. Use Streamlit secrets or environment variables only.
