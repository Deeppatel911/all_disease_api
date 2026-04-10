# ALL (Acute Lymphoblastic Leukemia) Disease Classification API

A FastAPI-based inference service that classifies microscopic blood smear images into ALL subtypes using a fine‑tuned DenseNet121 model (99.37% accuracy).

---

## Run Locally

### 1. Clone the repository
git clone https://github.com/Deeppatel911/all_disease_api.git
cd all_disease_api

### 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Start the API server
uvicorn app.main:app --reload

Swagger UI will be available at:
http://127.0.0.1:8000/docs

---

## 📸 Example API Request

### Endpoint
POST `/predict`

## Example Response
{
  "prediction": "Pro",
  "confidence": 0.9937
}

---

## Model Performance

| Model        | Accuracy |
|--------------|----------|
| DenseNet121  | **99.37%** |
| ResNet152    | 88.51% |

DenseNet121 is used in the deployed inference pipeline.

---

## Tech Stack
- FastAPI
- Uvicorn
- TensorFlow (DenseNet121)
- Python 3.10
- NumPy, Pillow, scikit-image

---

## Project Structure
all_disease_api/
├── app/  
│   ├── main.py  
│   ├── model/  
│   └── utils/  
├── requirements.txt  
├── Procfile  
└── README.md
