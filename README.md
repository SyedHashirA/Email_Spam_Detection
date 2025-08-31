# 📧 Spam Email PDF Classifier

A web-based application that allows users to **upload email messages in PDF format** and classifies them as **Spam** or **Non-Spam** using a trained machine learning model.  
The project integrates a **Python Flask backend** with a **React + Vite frontend**.

---

## 🚀 Features
- Upload any **PDF email document** for classification.
- Extracts text content using **PyPDF2**.
- Machine Learning model trained on the **TREC 2007 Public Spam Corpus**.
- Uses **TF-IDF vectorization** + **Logistic Regression** classifier.
- Returns classification result with **confidence score**.
- User-friendly **React frontend** with dark theme styling.
- REST API built with **Flask**.

---

## 📂 Project Structure
```
spam-pdf-classifier-starter/
│
├── backend/                # Flask backend
│   ├── train.py            # Training script for ML model
│   ├── app.py              # Flask API endpoints
│   ├── models/             # Saved model + metrics.json
│   └── data/               # Place dataset CSV here
│
├── frontend/               # React + Vite frontend
│   ├── src/                # React components
│   ├── public/             # Static assets
│   └── package.json
│
├── README.md               # Project documentation
└── report.pdf              # College project report (deliverable)
```

---

## ⚙️ Installation and Setup

### 1. Backend (Flask + ML)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
```

#### Download the Dataset
This project uses the **TREC 2007 Public Spam Corpus**.  
Download the preprocessed dataset from Kaggle:  
👉 [TREC 2007 Public Spam Dataset](https://www.kaggle.com/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset)

After downloading:
- Extract the CSV file.  
- Rename it to `spam.csv`.  
- Place it inside the `backend/data/` folder.  

#### Train the Model
```bash
python3 train.py --data ./data/spam.csv --model ./models/model.joblib --report ./models/metrics.json
```

#### Run the API
```bash
python3 app.py
```
Backend will start at: [http://localhost:5002](http://localhost:5002)  

---

### 2. Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

Frontend will run at: [http://localhost:5173](http://localhost:5173)  

---

## 📊 Dataset
- Source: [TREC 2007 Public Spam Corpus (Kaggle)](https://www.kaggle.com/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset)  
- Preprocessed into `spam.csv` with columns:
  - `label` → {0 = ham, 1 = spam}  
  - `text` → email message content  

### Example Rows
```csv
label,text
1,"Congratulations! You have won a FREE vacation. Click here to claim."
0,"Hi John, please find attached the updated project report."
```

---

## 📈 Results
After training with Logistic Regression:
- **Accuracy:** ~95%  
- **Precision:** ~94%  
- **Recall:** ~96%  
- **F1-score:** ~95%  

The model performs well in distinguishing spam from legitimate emails.

---

## 🖼️ Screenshots
Include the following in your report:
1. **Frontend (Upload page)** – PDF upload form + result display.  
2. **Backend (Terminal)** – running `python3 app.py` and showing logs when a PDF is classified.  
3. **Training Output** – terminal showing model metrics (accuracy, precision, recall).  

---

## 📌 Future Improvements
- Support for additional ML models (SVM, Random Forest, Deep Learning).
- Include multilingual spam detection.  
- Deploy on **Heroku/AWS/GCP** for cloud access.  
- Add continuous learning with user feedback.  

---

## 👨‍💻 Author
- **Syed Hashir Ahmed**  
- B.Tech CSE Student  
- [Your College Name]  

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![React](https://img.shields.io/badge/react-18.0-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0-black.svg)
