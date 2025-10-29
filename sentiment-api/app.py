import pickle
import re
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage
import json
import os


# TEXT CLEANING
_url = re.compile(r'https?://\S+|www\.\S+')


def clean_text(text: str) -> str:
    """Clean and preprocess text for model input."""
    t = _url.sub(' ', str(text).lower())
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    toks = [w for w in t.split() if len(w) >= 2]
    return " ".join(toks)


# MODEL LOADING FROM GCS
BUCKET_NAME = "model_mlops"
BLOB_PATH = "Team-14-v2.pickle"
METADATA_PATH = "Team-14-v2_metadata.json"
LOCAL_MODEL_PATH = "/tmp/model.pkl"
LOCAL_METADATA_PATH = "/tmp/metadata.json"


def load_model_from_gcs():
    """Download the latest production model and metadata from pipeline."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    # Load model
    blob_model = bucket.blob(BLOB_PATH)
    blob_model.download_to_filename(LOCAL_MODEL_PATH)
    
    with open(LOCAL_MODEL_PATH, "rb") as f:
        package = pickle.load(f)
    
    model = package["model"]
    vectorizer = package["vectorizer"]
    
    # Load metadata
    metadata = {}
    try:
        blob_meta = bucket.blob(METADATA_PATH)
        blob_meta.download_to_filename(LOCAL_METADATA_PATH)
        with open(LOCAL_METADATA_PATH, "r") as f:
            metadata = json.load(f)
        print("✓ Loaded metadata from GCS")
    except Exception as e:
        print(f"⚠ Could not load metadata file: {e}")
        metadata = {
            'current_version': package.get('version', 1),
            'current_accuracy': package.get('accuracy', 0.85),
            'current_timestamp': package.get('timestamp', datetime.now().isoformat()),
            'team': 'Team-14',
            'samples_trained': package.get('samples_trained', 5000),
            'features': package.get('features', 5000),
        }
    
    return model, vectorizer, metadata


# Load model on startup
print("🔄 Loading model from GCS...")
MODEL, VECTORIZER, METADATA = load_model_from_gcs()
print(f"✓ Loaded Model v{METADATA.get('current_version', '?')}")
print(f"✓ Accuracy: {METADATA.get('current_accuracy', '?')}")


# FASTAPI APP
app = FastAPI(
    title="Team 14 - Sentiment Analysis API",
    description="Production ML API powered by automated MLOps pipeline",
    version=str(METADATA.get('current_version', '1.0'))
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Review(BaseModel):
    text: str


@app.get("/")
def home():
    """Serve modern dark mode UI"""
    html_file = 'index.html'
    if os.path.exists(html_file):
        with open(html_file, 'r') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=get_default_html())


@app.get("/info")
def get_info():
    """Get comprehensive model metadata from pipeline."""
    version = METADATA.get('current_version', METADATA.get('version', 1))
    accuracy = METADATA.get('current_accuracy', METADATA.get('accuracy', 0.85))
    timestamp = METADATA.get('current_timestamp', METADATA.get('timestamp', datetime.now().isoformat()))
    
    response = {
        # Core metrics
        "version": version,
        "accuracy": float(accuracy),
        "samples_trained": METADATA.get('samples_trained', 5000),
        "timestamp": timestamp,
        
        # Performance metrics
        "precision": METADATA.get('precision'),
        "recall": METADATA.get('recall'),
        "f1_score": METADATA.get('f1_score'),
        
        # Model configuration
        "features": METADATA.get('features', 5000),
        "model_type": METADATA.get('model_type', 'LogisticRegression'),
        "vectorizer_type": METADATA.get('vectorizer_type', 'TfidfVectorizer'),
        "max_features": METADATA.get('max_features', 5000),
        "max_iterations": METADATA.get('max_iterations', 500),
        "training_duration": METADATA.get('training_duration_formatted'),
        
        # Dataset info
        "total_samples": METADATA.get('total_samples'),
        "samples_tested": METADATA.get('samples_tested'),
        "positive_samples": METADATA.get('positive_samples'),
        "negative_samples": METADATA.get('negative_samples'),
        "class_balance": METADATA.get('class_balance'),
        
        # Team info
        "team": METADATA.get('team', 'Team-14'),
        "pipeline_name": METADATA.get('pipeline_name', 'sentiment-pipeline-team14-final'),
    }
    
    # Remove None values
    return {k: v for k, v in response.items() if v is not None}


@app.post("/predict")
def predict(review: Review):
    """Predict sentiment for a given review."""
    cleaned = clean_text(review.text)
    vec = VECTORIZER.transform([cleaned])
    pred = MODEL.predict(vec)[0]
    prob = MODEL.predict_proba(vec)[0]
    
    return {
        "prediction": int(pred),
        "sentiment": "Positive" if pred == 1 else "Negative",
        "probability": float(prob[pred]),
        "confidence": f"{prob[pred]*100:.2f}%",
        "cleaned_text": cleaned,
        "model_version": METADATA.get('current_version', METADATA.get('version', 1)),
        "model_accuracy": METADATA.get('current_accuracy', METADATA.get('accuracy'))
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "vectorizer_loaded": VECTORIZER is not None,
        "metadata_loaded": bool(METADATA),
        "model_version": METADATA.get('current_version', METADATA.get('version', 1))
    }


def get_default_html():
    """Return simplified HTML WITHOUT confusion matrix"""
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Team 14 - Sentiment Analysis</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter','Segoe UI',sans-serif;background:linear-gradient(135deg,#0a0a0a 0%,#1a1a2e 100%);color:#e0e0e0;min-height:100vh;padding:20px}
.container{max-width:1100px;margin:0 auto;padding:40px 20px}
header{text-align:center;margin-bottom:50px}
h1{font-size:2.5rem;background:linear-gradient(90deg,#00d4ff,#00ff88);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px;font-weight:700}
.subtitle{color:#888;font-size:1.1rem}
.card{background:rgba(255,255,255,0.05);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:30px;margin-bottom:30px;box-shadow:0 8px 32px rgba(0,0,0,0.3)}
.model-info{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:20px}
.info-item{background:rgba(0,212,255,0.1);padding:15px;border-radius:10px;border-left:3px solid #00d4ff;transition:all 0.3s}
.info-item:hover{background:rgba(0,212,255,0.15);transform:translateY(-2px)}
.info-label{font-size:0.85rem;color:#888;text-transform:uppercase;letter-spacing:1px}
.info-value{font-size:1.3rem;font-weight:600;color:#00d4ff;margin-top:5px}
.expand-btn{background:rgba(0,212,255,0.2);border:1px solid rgba(0,212,255,0.3);color:#00d4ff;padding:12px 24px;border-radius:8px;cursor:pointer;font-size:0.95rem;font-weight:600;transition:all 0.3s;margin-top:20px;width:100%}
.expand-btn:hover{background:rgba(0,212,255,0.3);border-color:#00d4ff}
.details{display:none;margin-top:20px;padding:20px;background:rgba(0,0,0,0.3);border-radius:12px;border:1px solid rgba(255,255,255,0.1)}
.details.show{display:block;animation:slideDown 0.3s ease-out}
.detail-section{margin-bottom:25px}
.detail-section h3{color:#00ff88;font-size:1.2rem;margin-bottom:15px;border-bottom:2px solid rgba(0,255,136,0.3);padding-bottom:8px}
.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.detail-item{background:rgba(255,255,255,0.03);padding:12px;border-radius:8px;border-left:2px solid rgba(0,212,255,0.5)}
.detail-label{font-size:0.8rem;color:#aaa;text-transform:uppercase}
.detail-value{font-size:1.1rem;color:#fff;margin-top:4px;font-weight:500}
textarea{width:100%;min-height:150px;background:rgba(255,255,255,0.05);border:2px solid rgba(255,255,255,0.1);border-radius:12px;padding:15px;color:#e0e0e0;font-size:1rem;font-family:inherit;resize:vertical;transition:all 0.3s}
textarea:focus{outline:none;border-color:#00d4ff;box-shadow:0 0 20px rgba(0,212,255,0.2)}
button{background:linear-gradient(90deg,#00d4ff,#00ff88);color:#000;border:none;padding:15px 40px;font-size:1.1rem;font-weight:600;border-radius:12px;cursor:pointer;transition:transform 0.2s,box-shadow 0.2s;width:100%;margin-top:15px}
button:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(0,212,255,0.3)}
button:active{transform:translateY(0)}
.result{margin-top:30px;padding:20px;border-radius:12px;border-left:4px solid;animation:slideIn 0.3s ease-out}
.result.positive{background:rgba(0,255,136,0.1);border-color:#00ff88}
.result.negative{background:rgba(255,64,129,0.1);border-color:#ff4081}
.result-label{font-size:0.9rem;color:#888;text-transform:uppercase;letter-spacing:1px}
.result-value{font-size:2rem;font-weight:700;margin-top:5px}
.confidence{font-size:1.2rem;margin-top:10px;opacity:0.8}
@keyframes slideIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideDown{from{opacity:0;max-height:0}to{opacity:1;max-height:2000px}}
.hidden{display:none}
.loading{display:inline-block;width:20px;height:20px;border:3px solid rgba(255,255,255,0.3);border-radius:50%;border-top-color:#00d4ff;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body>
<div class="container"><header><h1>Sentiment Analysis API</h1><p class="subtitle">Powered by Team 14 MLOps Pipeline</p></header>
<div class="card"><h2 style="margin-bottom:20px;color:#00d4ff">Model Information</h2>
<div class="model-info">
<div class="info-item"><div class="info-label">Version</div><div class="info-value" id="version">Loading...</div></div>
<div class="info-item"><div class="info-label">Accuracy</div><div class="info-value" id="accuracy">Loading...</div></div>
<div class="info-item"><div class="info-label">Samples Trained</div><div class="info-value" id="samples">Loading...</div></div>
<div class="info-item"><div class="info-label">Last Updated</div><div class="info-value" id="timestamp">Loading...</div></div>
</div>
<button class="expand-btn" onclick="toggleDetails()">Show Detailed Metrics</button>
<div class="details" id="detailsSection">
<div class="detail-section">
<h3>Performance Metrics</h3>
<div class="detail-grid">
<div class="detail-item"><div class="detail-label">Precision</div><div class="detail-value" id="precision">-</div></div>
<div class="detail-item"><div class="detail-label">Recall</div><div class="detail-value" id="recall">-</div></div>
<div class="detail-item"><div class="detail-label">F1-Score</div><div class="detail-value" id="f1">-</div></div>
<div class="detail-item"><div class="detail-label">Training Duration</div><div class="detail-value" id="duration">-</div></div>
</div>
</div>
<div class="detail-section">
<h3>Dataset Information</h3>
<div class="detail-grid">
<div class="detail-item"><div class="detail-label">Total Samples</div><div class="detail-value" id="total">-</div></div>
<div class="detail-item"><div class="detail-label">Test Samples</div><div class="detail-value" id="testSamples">-</div></div>
<div class="detail-item"><div class="detail-label">Positive Samples</div><div class="detail-value" id="posSamples">-</div></div>
<div class="detail-item"><div class="detail-label">Negative Samples</div><div class="detail-value" id="negSamples">-</div></div>
<div class="detail-item"><div class="detail-label">Class Balance</div><div class="detail-value" id="balance">-</div></div>
<div class="detail-item"><div class="detail-label">Features</div><div class="detail-value" id="features">-</div></div>
</div>
</div>
<div class="detail-section">
<h3>Model Configuration</h3>
<div class="detail-grid">
<div class="detail-item"><div class="detail-label">Algorithm</div><div class="detail-value" id="algo">-</div></div>
<div class="detail-item"><div class="detail-label">Vectorizer</div><div class="detail-value" id="vec">-</div></div>
<div class="detail-item"><div class="detail-label">Max Features</div><div class="detail-value" id="maxFeat">-</div></div>
<div class="detail-item"><div class="detail-label">Max Iterations</div><div class="detail-value" id="maxIter">-</div></div>
</div>
</div>
</div>
</div>
<div class="card"><h2 style="margin-bottom:20px;color:#00d4ff">Test the Model</h2>
<textarea id="reviewText" placeholder="Enter a movie review here...

Example: This movie was absolutely amazing! The cinematography was stunning and the acting was superb. A masterpiece!"></textarea>
<button onclick="predict()"><span id="buttonText">Analyze Sentiment</span><span id="buttonLoading" class="loading hidden"></span></button>
<div id="result" class="hidden"></div></div></div>
<script>
fetch('/info').then(r=>r.json()).then(data=>{
document.getElementById('version').textContent='v'+data.version;
document.getElementById('accuracy').textContent=(data.accuracy*100).toFixed(2)+'%';
document.getElementById('samples').textContent=data.samples_trained.toLocaleString();
document.getElementById('timestamp').textContent=new Date(data.timestamp).toLocaleDateString();
document.getElementById('precision').textContent=data.precision?(data.precision*100).toFixed(2)+'%':'N/A';
document.getElementById('recall').textContent=data.recall?(data.recall*100).toFixed(2)+'%':'N/A';
document.getElementById('f1').textContent=data.f1_score?(data.f1_score*100).toFixed(2)+'%':'N/A';
document.getElementById('duration').textContent=data.training_duration||'N/A';
document.getElementById('total').textContent=data.total_samples?.toLocaleString()||'N/A';
document.getElementById('testSamples').textContent=data.samples_tested?.toLocaleString()||'N/A';
document.getElementById('posSamples').textContent=data.positive_samples?.toLocaleString()||'N/A';
document.getElementById('negSamples').textContent=data.negative_samples?.toLocaleString()||'N/A';
document.getElementById('balance').textContent=data.class_balance?(data.class_balance*100).toFixed(1)+'%':'N/A';
document.getElementById('features').textContent=data.features?.toLocaleString()||'N/A';
document.getElementById('algo').textContent=data.model_type||'N/A';
document.getElementById('vec').textContent=data.vectorizer_type||'N/A';
document.getElementById('maxFeat').textContent=data.max_features?.toLocaleString()||'N/A';
document.getElementById('maxIter').textContent=data.max_iterations?.toLocaleString()||'N/A';
});
function toggleDetails(){
const details=document.getElementById('detailsSection');
const btn=event.target;
if(details.classList.contains('show')){
details.classList.remove('show');
btn.textContent='Show Detailed Metrics';
}else{
details.classList.add('show');
btn.textContent='Hide Detailed Metrics';
}}
async function predict(){
const text=document.getElementById('reviewText').value;
if(!text.trim()){alert('Please enter a review!');return;}
document.getElementById('buttonText').classList.add('hidden');
document.getElementById('buttonLoading').classList.remove('hidden');
document.getElementById('result').classList.add('hidden');
try{
const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text})});
const data=await response.json();
const resultDiv=document.getElementById('result');
const sentiment=data.prediction===1?'Positive':'Negative';
const probability=(data.probability*100).toFixed(2);
const color=data.prediction===1?'positive':'negative';
resultDiv.className='result '+color;
resultDiv.innerHTML='<div class="result-label">Sentiment</div><div class="result-value">'+sentiment+'</div><div class="confidence">Confidence: '+probability+'%</div>';
resultDiv.classList.remove('hidden');
}catch(e){alert('Error: '+e.message);}finally{
document.getElementById('buttonText').classList.remove('hidden');
document.getElementById('buttonLoading').classList.add('hidden');
}}
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
 # Add blank line
