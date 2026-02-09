# Electricity Supply Band Predictor API

A FastAPI-based REST API for predicting electricity supply bands in the Nigerian power grid.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Run the API Locally
```bash
python api.py
```

The API will start on `http://localhost:8000`

## API Documentation

Once the API is running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

## Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information.

### 2. Health Check
```
GET /health
```
Returns the health status and whether the model is loaded.

### 3. Predict Supply Band
```
POST /predict
```

**Request Body:**
```json
{
  "disco": "IKEDC",
  "zone": "Urban",
  "feeder_age": 5.0,
  "transformer_issue": false
}
```

**Response:**
```json
{
  "supply_band": "A",
  "confidence": 0.8542
}
```

### Parameters:
- **disco** (string, required): Distribution Company
  - Options: IKEDC, AEDC, EKEDC, KEDCO, IBEDC
- **zone** (string, required): Area Type
  - Options: Urban, Suburban, Rural
- **feeder_age** (float, required): Age of feeder in years
- **transformer_issue** (boolean, required): Whether there's a transformer issue

### Supply Bands:
- **A**: ≥ 20 hours/day
- **B**: 16-19 hours/day
- **C**: 12-15 hours/day
- **D**: 8-11 hours/day
- **E**: < 8 hours/day

## Testing the API

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"disco": "IKEDC", "zone": "Urban", "feeder_age": 5.0, "transformer_issue": false}'
```

### Using Python Client
```bash
python test_api_client.py
```

## Deployment Options

### Option 1: Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements_api.txt
CMD ["python", "api.py"]
```

Build and run:
```bash
docker build -t electricity-api .
docker run -p 8000:8000 electricity-api
```

### Option 2: Cloud Platforms

**Heroku:**
```bash
heroku create your-app-name
git push heroku main
```

**AWS Lambda (with AWS API Gateway):**
- Package the application using Mangum
- Deploy via SAM or Serverless Framework

**Azure App Service:**
```bash
az webapp up --name your-app-name --runtime "PYTHON:3.11"
```

**Google Cloud Run:**
```bash
gcloud run deploy electricity-api --source .
```

## Model Persistence

The API automatically:
- Trains and saves the model on first run
- Loads the saved model on subsequent runs
- Saves model artifacts in:
  - `electricity_model.pkl`
  - `label_encoder.pkl`
  - `feature_names.pkl`

## Architecture

```
elec.py (Core Model)
    ↓
api.py (FastAPI Server)
    ↓
test_api_client.py (Client Tests)
```

## Performance

- Model: XGBoost Classifier
- Training Data: 2000 feeders
- Classes: 5 (A, B, C, D, E)
- Response Time: < 100ms per prediction

## Troubleshooting

**Port already in use:**
```bash
python api.py --port 8001
```

**Model not loading:**
- Delete `.pkl` files to retrain
- Check `elec.py` dependencies are installed

**Import errors:**
- Ensure `elec.py` is in the same directory
- Run `pip install -r requirements_api.txt`
