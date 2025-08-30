# 📧 Spam Email PDF Classifier

A web-based application that allows users to **upload email messages in PDF format** and classifies them as **Spam** or **Non-Spam** using a trained machine learning model.  
The project integrates a **Python FastAPI backend** with a **React + Vite frontend**.

---

## 🚀 Features
- Upload any **PDF email document** for classification.
- Extracts text content using **PyPDF2**.
- Machine Learning model trained on the **TREC 2007 Public Spam Corpus**.
- Uses **TF-IDF vectorization** + **Logistic Regression** classifier.
- Returns classification result with **confidence score**.
- User-friendly **React frontend** with dark theme styling.
- REST API built with **FastAPI**.

---

## 📂 Project Structure
spam-pdf-classifier-starter/
│
├── backend/                # FastAPI backend
│   ├── train.py            # Training script for ML model
│   ├── main.py             # API endpoints
│   ├── models/             # Saved model + metrics.json
│   └── data/               # Dataset (CSV files)
│
├── frontend/               # React + Vite frontend
│   ├── src/                # React components
│   ├── public/             # Static assets
│   └── package.json
│
├── README.md               # Project documentation

---

## ⚙️ Installation and Setup

1. Backend (FastAPI + ML)
cd backend
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt

Train the model
python train.py --data ./data/spam.csv --model ./models/model.joblib --report ./models/metrics.json

Run the API
python3 app.py

Backend will start at: http://localhost:5002 


2. Frontend (React + Vite)
cd frontend
npm install
npm run dev

	•	Dataset used: TREC 2007 Public Spam Corpus
	•	Preprocessed into spam.csv with columns:
	•	label → {0 = ham, 1 = spam}
	•	text → email message content

  📈 Results

After training with Logistic Regression:
	•	Accuracy: ~95%
	•	Precision: ~94%
	•	Recall: ~96%
	•	F1-score: ~95%

The model performs well in distinguishing spam from legitimate emails.

📌 Future Improvements
	•	Support for additional ML models (SVM, Random Forest, Deep Learning).
	•	Include multilingual spam detection.
	•	Deploy on Heroku/AWS/GCP for cloud access.
	•	Add continuous learning with user feedback.

👨‍💻 Author
	•	Syed Hashir Ahmed
	•	B.Tech CSE Student



