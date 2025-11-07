# 🔮 DiabRisk AI: Early Diabetes Risk Predictor

DiabRisk AI is a full-stack application for **early diabetes risk prediction**. It features a robust **ML pipeline (85.2% AUC)** built on NHANES data, an **explainable AI (XAI)** engine, and a **modern React frontend**.

---

## 🚀 Key Features

- 🤖 **AI-Powered Prediction:** Uses a validated XGBoost model (ROC AUC: 0.852, Accuracy: 84.2%) to generate a real-time risk score.
- 🧠 **Explainable AI (XAI):** Integrates SHAP to show users *why* they received their score, detailing the top contributing risk factors for their specific prediction.
- 💡 **Personalized Advice:** Generates actionable lifestyle recommendations based on user inputs.
- 📊 **Data Dashboard:** A separate `/dashboard` route provides a deep dive into the model's metrics, feature importance, and performance comparisons.

---

## 🧬 The ML Pipeline: A Focus on Robustness

This project's core is a **machine learning pipeline that explicitly avoids data leakage.**

### ❌ The Problem:
Many public models are “cheated” by training on diagnostic data (e.g., A1c, Glucose levels) to “predict” diabetes.  
This results in falsely high 99%+ accuracy and makes the model useless for real-world screening.

### ✅ The Solution:
This model is trained only on **non-diagnostic risk factors** (e.g., BMI, age, blood pressure, lipids), simulating a realistic screening scenario.  
The **85.2% AUC** represents an honest, robust, and deployable model that is competitive with existing clinical research.

---

## ⚙️ Getting Started

Follow these steps to run the complete application on your local machine.

### 🧩 Prerequisites
- Python 3.11+
- Node.js v18+

---

## 🧠 Backend (FastAPI Server)

The backend server runs on **http://127.0.0.1:5000**.

# 1️⃣ Navigate to the backend folder
```bash
cd backend
```
# 2️⃣ Install Python dependencies
```bash
pip install -r requirements.txt
```
# 3️⃣ Run the ML pipeline to generate models (only need to do this once)
```bash
python src/ml/pipeline.py
```
# 4️⃣ Run the FastAPI server
```bash
uvicorn app:app --reload --port 5000
```

---

## 💻 Frontend (Next.js App)

The frontend runs on **http://localhost:3000**.

# 1️⃣ In a new terminal, navigate to the frontend folder
```bash
cd frontend
```
# 2️⃣ Install Node.js dependencies
```bash
npm install
```
# 3️⃣ Run the development server
```bash
npm run dev
```

Then open **http://localhost:3000** in your browser to use the application.

---

## 📄 License

This project is licensed under the **MIT License**.
