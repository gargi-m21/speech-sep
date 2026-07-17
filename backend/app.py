import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.services.inference import SpeakerSeparationService

# Initialize FastAPI App
app = FastAPI(
    title="MTC-Net API",
    description="REST API for speech separation using MTC-Net",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize Service
try:
    separation_service = SpeakerSeparationService(outputs_dir=OUTPUTS_DIR)
except Exception as e:
    print(f"Error loading separation service: {e}")
    separation_service = None

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model": "MTC-Net Speech Separation Service",
        "device": "cuda" if separation_service and separation_service.device.type == "cuda" else "cpu"
    }

@app.post("/separate")
async def separate_audio(
    file: UploadFile = File(...),
    num_speakers: str = Form("auto")
):
    if not separation_service:
        raise HTTPException(
            status_code=500,
            detail="Separation service is not loaded/initialized properly."
        )

    # Validate file extension
    filename = file.filename
    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .wav file."
        )

    # Generate unique temporary file name
    temp_filename = f"{uuid.uuid4()}_{filename}"
    temp_file_path = os.path.join(UPLOADS_DIR, temp_filename)

    # Save uploaded file
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # Run separation
    try:
        result = separation_service.separate(temp_file_path, num_speakers=num_speakers)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Speech separation failed: {str(e)}"
        )
    finally:
        # Clean up original uploaded file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
