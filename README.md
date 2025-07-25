# Persian License Plate Recognition API

A production-ready Flask API for Persian car license plate recognition and anonymization (blur/white effect), with Docker support.

## Features
- Upload an image and get the recognized license plate number
- Apply blur or white effect to the plate area
- Download the processed image (auto-deletes after a set time)
- Configurable via `.env` file
- Production-ready Dockerfile and docker-compose

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sayheymirza/persian-license-plate-recognition.git
cd persian-license-plate-recognition
```

### 2. Build and run with Docker Compose
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8080` by default.

### 3. API Usage
#### POST `/process-plate`
- **file**: image file (form-data)
- **effect**: `blur` or `white` (default: `blur`)

**Response:**
```
{
  "ok": true,
  "status": 200,
  "meta": { "took": 0.45 },
  "number": "۱۲۳الف۴۵",
  "url": "/download/<filename>"
}
```

#### GET `/download/<filename>`
Download the processed image (valid for a limited time).

## Configuration
Edit `.env` for custom settings:
```
UPLOAD_FOLDER=public
FILE_LIFETIME=60
PORT=8080
DEBUG=False
```

## Development
- Install requirements: `pip install -r requirements.txt`
- Run locally: `python app.py`

## License
MIT