# AI-Powered Fraud Detection and Risk Prediction System Using Ensemble Learning and LLMs

An interactive Streamlit demonstration that estimates transaction fraud risk using an ensemble machine-learning model and explains the result in analyst-friendly language.

## What is included?

- Overview dashboard with transaction volume, known demo fraud labels, risk bands, and channel analysis.
- Interactive transaction assessment form.
- Searchable and filterable transaction explorer.
- Model lab with ROC-AUC, precision, recall, F1, confusion matrix, and feature importance.
- Optional Gemini-powered natural-language explanation with a deterministic local fallback.
- A project guide that explains the problem, workflow, technologies, and limitations.

## Technology stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Gemini API (optional).

## Run on Windows

1. Extract the ZIP file.
2. Open the `fraud_risk_streamlit` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   python -m venv venv
   ```

5. Activate it:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

6. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Start the app:

   ```powershell
   streamlit run app.py
   ```

8. Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser.

### One-click Windows option

Double-click `run_windows.bat`. It creates the virtual environment if needed, installs dependencies, and launches Streamlit.

## Enable Gemini explanations (optional)

1. Copy `.env.example` to a new file named `.env`.
2. Add your Gemini API key after `GEMINI_API_KEY=`.
3. Restart Streamlit.
4. Open **Analyze a transaction** and click **Generate explanation with Gemini**.

The app works without an API key. In that case, it provides a local rule-based explanation. The LLM does not generate the model score; it only explains the model output.

## Model approach

The project uses a soft-voting ensemble of:
- Random Forest
- Gradient Boosting

Both estimators produce class probabilities. Soft voting averages their probabilities. A held-out synthetic test split is used to calculate ROC-AUC, precision, recall, F1, and a confusion matrix.

## Data and limitations

All transaction data and fraud labels are synthetically generated when the app starts. The generated labels are based on documented risk-related feature patterns plus randomness. This makes the project reproducible and self-contained, but it does **not** represent real financial data or real fraud prevalence.

This is an educational prototype only. Do not use it to approve, reject, or block real payments. A production-grade system requires real representative data, privacy and security controls, threshold calibration, fairness assessment, drift monitoring, audit trails, and human review.
