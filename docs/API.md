# API Documentation

## Starting the API

```bash
python api/main.py
# or
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

### Health Check
```bash
GET /health
```

### Generate SAR
```bash
POST /api/v1/sar/generate
```

### KYC Assessment
```bash
POST /api/v1/kyc/assess
```

### Transaction Analysis
```bash
POST /api/v1/transaction/analyze
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for Swagger UI.
