import os
import uuid
import shutil
import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from pydantic import BaseModel
from typing import Optional, List

BASE_DIR = Path(__file__).parent
app = FastAPI()

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# Directory to store uploaded images
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Database configuration: use DATABASE_URL env var for MariaDB, fallback to sqlite for local dev
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{BASE_DIR / 'data.db'}"

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), unique=True, nullable=False)
    original_name = Column(String(255), nullable=False)
    content_type = Column(String(120), nullable=False)
    device_id = Column(String(128), nullable=True, index=True)
    latitude = Column(String(64), nullable=True)
    longitude = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Telemetry(Base):
    __tablename__ = "telemetry"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(128), nullable=False, index=True)
    latitude = Column(String(64), nullable=False)
    longitude = Column(String(64), nullable=False)
    altitude = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)

# Mount uploaded images folder so files are served at /images/files/<filename>
app.mount("/images/files", StaticFiles(directory=IMAGES_DIR), name="uploaded_images")


@app.get("/", response_class=HTMLResponse)
def home():
    html_file = BASE_DIR / "index.html"
    return html_file.read_text()


@app.get("/health")
def health():
    return {"status": "Server is running"}


@app.post("/upload")
def upload_image(file: UploadFile = File(...), device_id: Optional[str] = Form(None), latitude: Optional[float] = Form(None), longitude: Optional[float] = Form(None)):
    # Save file to disk with a uuid filename
    try:
        ext = Path(file.filename).suffix
        unique_name = f"{uuid.uuid4().hex}{ext}"
        dest = IMAGES_DIR / unique_name
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # persist metadata to DB; attach provided or latest telemetry if available
        db = SessionLocal()
        try:
            # if lat/lon not provided but device_id is, try to fetch latest telemetry for device
            if (latitude is None or longitude is None) and device_id:
                latest = db.query(Telemetry).filter(Telemetry.device_id == device_id).order_by(Telemetry.created_at.desc()).first()
                if latest:
                    if latitude is None:
                        latitude = float(latest.latitude)
                    if longitude is None:
                        longitude = float(latest.longitude)

            img = Image(filename=unique_name, original_name=file.filename, content_type=file.content_type or "application/octet-stream", device_id=device_id, latitude=(str(latitude) if latitude is not None else None), longitude=(str(longitude) if longitude is not None else None))
            db.add(img)
            db.commit()
            db.refresh(img)
        finally:
            db.close()

        return {"id": img.id, "url": f"/images/files/{unique_name}", "original_name": img.original_name, "device_id": img.device_id, "latitude": img.latitude, "longitude": img.longitude}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TelemetryIn(BaseModel):
    device_id: str
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    timestamp: Optional[str] = None


class TelemetryOut(BaseModel):
    id: int
    device_id: str
    latitude: float
    longitude: float
    altitude: Optional[float]
    created_at: str


@app.post("/telemetry", response_model=TelemetryOut)
def post_telemetry(payload: TelemetryIn):
    db = SessionLocal()
    try:
        t = Telemetry(device_id=payload.device_id, latitude=str(payload.latitude), longitude=str(payload.longitude), altitude=(str(payload.altitude) if payload.altitude is not None else None))
        db.add(t)
        db.commit()
        db.refresh(t)
        return TelemetryOut(id=t.id, device_id=t.device_id, latitude=float(t.latitude), longitude=float(t.longitude), altitude=(float(t.altitude) if t.altitude is not None else None), created_at=t.created_at.isoformat())
    finally:
        db.close()


@app.get("/telemetry", response_model=List[TelemetryOut])
def get_telemetry(device_id: Optional[str] = None, limit: int = 100):
    db = SessionLocal()
    try:
        q = db.query(Telemetry)
        if device_id:
            q = q.filter(Telemetry.device_id == device_id)
        q = q.order_by(Telemetry.created_at.desc()).limit(limit)
        rows = q.all()
        return [TelemetryOut(id=r.id, device_id=r.device_id, latitude=float(r.latitude), longitude=float(r.longitude), altitude=(float(r.altitude) if r.altitude is not None else None), created_at=r.created_at.isoformat()) for r in rows]
    finally:
        db.close()


@app.get("/telemetry/{device_id}/latest", response_model=TelemetryOut)
def get_latest_for_device(device_id: str):
    db = SessionLocal()
    try:
        r = db.query(Telemetry).filter(Telemetry.device_id == device_id).order_by(Telemetry.created_at.desc()).first()
        if not r:
            raise HTTPException(status_code=404, detail="No telemetry for device")
        return TelemetryOut(id=r.id, device_id=r.device_id, latitude=float(r.latitude), longitude=float(r.longitude), altitude=(float(r.altitude) if r.altitude is not None else None), created_at=r.created_at.isoformat())
    finally:
        db.close()


@app.get("/images")
def list_images():
    db = SessionLocal()
    images = db.query(Image).order_by(Image.created_at.desc()).all()
    db.close()
    return [{"id": i.id, "original_name": i.original_name, "url": f"/images/files/{i.filename}", "created_at": i.created_at.isoformat(), "device_id": i.device_id, "latitude": (float(i.latitude) if i.latitude is not None else None), "longitude": (float(i.longitude) if i.longitude is not None else None)} for i in images]


@app.get("/images/{image_id}")
def get_image(image_id: int):
    db = SessionLocal()
    img = db.query(Image).filter(Image.id == image_id).first()
    db.close()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    file_path = IMAGES_DIR / img.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(file_path, media_type=img.content_type, filename=img.original_name)
