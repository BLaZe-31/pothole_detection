from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2, io, os
from fastapi.responses import JSONResponse
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html


@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    return JSONResponse(content={"status": "alive"})

@app.api_route("/", methods=["GET", "HEAD"])
async def read_index():
    return FileResponse("index.html")



# Static files (if needed for CSS/JS)
#app.mount("/static", StaticFiles(directory="."), name="static")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("runs/detect/civic_issue_model/weights/best.pt")

@app.post("/detect")
@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    results = model.predict(source=file_path, conf=0.25)
    annotated_image = results[0].plot()

    success, buffer = cv2.imencode(".jpg", annotated_image)
    if not success:
        return {"error": "Failed to encode image"}

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
