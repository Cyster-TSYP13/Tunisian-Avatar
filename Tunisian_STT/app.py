from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from vosk import Model, KaldiRecognizer
import wave
import json
import io
import time
import os
import tempfile

app = FastAPI(
    title="LinTO ASR Arabic Tunisia STT API",
    description="Speech-to-Text API for Tunisian Arabic dialect using Vosk",
    version="0.1.0"
)

# Global model instance (loaded once at startup)
model = None
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/STT_Tun_Model")

@app.on_event("startup")
async def load_model():
    """Load the Vosk model at startup"""
    global model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = Model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "LinTO ASR Arabic Tunisia",
        "version": "0.1.0",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint for Azure/Kubernetes"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text
    
    Args:
        audio: WAV audio file (mono, PCM format)
    
    Returns:
        JSON response with transcript and processing time
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not audio.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400, 
            detail="Only WAV files are supported"
        )
    
    start_time = time.time()
    
    try:
        # Read the uploaded file
        audio_bytes = await audio.read()
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Process the audio file
            with wave.open(tmp_path, "rb") as wf:
                # Validate audio format
                if wf.getnchannels() != 1:
                    raise HTTPException(
                        status_code=400, 
                        detail="Audio must be mono (1 channel)"
                    )
                if wf.getsampwidth() != 2:
                    raise HTTPException(
                        status_code=400, 
                        detail="Audio must be 16-bit PCM"
                    )
                if wf.getcomptype() != "NONE":
                    raise HTTPException(
                        status_code=400, 
                        detail="Audio must be uncompressed PCM"
                    )
                
                # Create recognizer and transcribe
                rec = KaldiRecognizer(model, wf.getframerate())
                rec.AcceptWaveform(wf.readframes(wf.getnframes()))
                res = rec.FinalResult()
                result_dict = json.loads(res)
                transcript = result_dict.get("text", "")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        end_time = time.time()
        latency = end_time - start_time
        
        return JSONResponse(content={
            "success": True,
            "transcript": transcript,
            "latency_seconds": round(latency, 2),
            "filename": audio.filename
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing audio: {str(e)}"
        )

@app.post("/transcribe/streaming")
async def transcribe_streaming(audio: UploadFile = File(...)):
    """
    Transcribe audio file with streaming results (partial + final)
    
    Args:
        audio: WAV audio file (mono, PCM format)
    
    Returns:
        JSON response with partial and final transcripts
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not audio.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400, 
            detail="Only WAV files are supported"
        )
    
    start_time = time.time()
    
    try:
        audio_bytes = await audio.read()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            with wave.open(tmp_path, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                    raise HTTPException(
                        status_code=400, 
                        detail="Audio must be WAV format mono PCM"
                    )
                
                rec = KaldiRecognizer(model, wf.getframerate())
                
                # Process in chunks for partial results
                partial_results = []
                chunk_size = 4000
                
                while True:
                    data = wf.readframes(chunk_size)
                    if len(data) == 0:
                        break
                    
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if result.get("text"):
                            partial_results.append(result["text"])
                
                # Get final result
                final_result = json.loads(rec.FinalResult())
                final_transcript = final_result.get("text", "")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        end_time = time.time()
        latency = end_time - start_time
        
        return JSONResponse(content={
            "success": True,
            "final_transcript": final_transcript,
            "partial_results": partial_results,
            "latency_seconds": round(latency, 2),
            "filename": audio.filename
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing audio: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
