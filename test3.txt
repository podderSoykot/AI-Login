from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
import cv2
import numpy as np
import face_recognition
import base64
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import threading
import queue
import time
import uvicorn
import os
from fastapi import Form
from fastapi.responses import JSONResponse
import traceback
import logging
from pydantic import BaseModel
import faiss
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System startup time
STARTUP_TIME = datetime.utcnow()

# ───── Optional: Improve H264 decoder handling ─────
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"

# ─── Database Setup ─────────────────────────────────────────────
Base = declarative_base()
engine = create_engine('sqlite:///face_auth.db')
Session = sessionmaker(bind=engine)

# Constants
FAISS_INDEX_PATH = "faiss_index.bin"
USER_MAPPING_PATH = "user_mapping.pkl"
FACE_DISTANCE_THRESHOLD = 0.6  # Threshold for face matching
BATCH_SIZE = 100  # Number of images to process in batch registration

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    face_encoding = Column(LargeBinary, nullable=False)
    registered_photo_small = Column(LargeBinary, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "face_encoding": base64.b64encode(self.face_encoding).decode('utf-8') if self.face_encoding else None,
            "registered_photo_small": base64.b64encode(self.registered_photo_small).decode('utf-8') if self.registered_photo_small else None
        }

class LoginHistory(Base):
    __tablename__ = 'login_history'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    name = Column(String(100), nullable=False)
    login_time = Column(DateTime, default=datetime.utcnow)
    registered_image = Column(LargeBinary, nullable=False)
    live_capture = Column(LargeBinary, nullable=False)
    status = Column(String(20), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "login_time": self.login_time.isoformat() if self.login_time else None,
            "registered_image": base64.b64encode(self.registered_image).decode('utf-8') if self.registered_image else None,
            "live_capture": base64.b64encode(self.live_capture).decode('utf-8') if self.live_capture else None,
            "status": self.status
        }

Base.metadata.create_all(engine)


# ─── FastAPI Setup ─────────────────────────────────────────────
app = FastAPI(title="Face Authentication System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

# ─── Camera RTSP Stream Setup ──────────────────────────────────
rtsp_url = "rtsp://admin:qwq1234.@192.168.8.110/channel=1/subtype=0"
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

def camera_thread():
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Failed to open RTSP stream")
        return
    logger.info("RTSP stream opened successfully")

    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if ret and frame is not None:
                if not frame_queue.empty():
                    frame_queue.get_nowait()
                frame_queue.put(frame)
            else:
                logger.warning("Frame read failed, retrying...")
        except cv2.error as e:
            logger.error(f"OpenCV error: {str(e)}")
        except Exception as ex:
            logger.error(f"Unexpected error: {ex}")
        time.sleep(0.03)

    cap.release()


threading.Thread(target=camera_thread, daemon=True).start()

def get_latest_frame(wait_time=5):
    """Wait up to wait_time seconds for the first frame"""
    start = time.time()
    while time.time() - start < wait_time:
        try:
            return frame_queue.get(timeout=1)
        except queue.Empty:
            logger.warning("Waiting for RTSP frame...")
    logger.error("Timeout: No frame received from RTSP stream")
    return None

# FAISS index for face encodings
face_index = None
user_mapping = {}  # Maps index to user_id

def save_faiss_index():
    """Save FAISS index and user mapping to disk"""
    try:
        if face_index is not None:
            # Save FAISS index
            faiss.write_index(face_index, FAISS_INDEX_PATH)
            # Save user mapping
            with open(USER_MAPPING_PATH, 'wb') as f:
                pickle.dump(user_mapping, f)
            logger.info("Saved FAISS index and user mapping to disk")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {str(e)}")
        logger.error(traceback.format_exc())

def load_faiss_index():
    """Load FAISS index and user mapping from disk"""
    global face_index, user_mapping
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(USER_MAPPING_PATH):
            # Load FAISS index
            face_index = faiss.read_index(FAISS_INDEX_PATH)
            # Load user mapping
            with open(USER_MAPPING_PATH, 'rb') as f:
                user_mapping = pickle.load(f)
            logger.info("Loaded FAISS index and user mapping from disk")
            return True
        return False
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def initialize_faiss():
    """Initialize FAISS index for face recognition"""
    global face_index
    # Create a FAISS index for L2 distance
    dimension = 128  # Face encoding dimension
    face_index = faiss.IndexFlatL2(dimension)
    return face_index

def update_faiss_index():
    """Update FAISS index with all user face encodings"""
    global face_index, user_mapping
    try:
        session = Session()
        users = session.query(User).all()
        
        if not users:
            return
        
        # Initialize FAISS index if not exists
        if face_index is None:
            face_index = initialize_faiss()
        
        # Clear existing index and mapping
        face_index.reset()
        user_mapping.clear()
        
        # Prepare data for FAISS
        encodings = []
        for idx, user in enumerate(users):
            try:
                encoding = np.frombuffer(user.face_encoding, dtype=np.float32)
                encodings.append(encoding)
                user_mapping[idx] = user.id
            except Exception as e:
                logger.error(f"Error processing encoding for user {user.id}: {str(e)}")
                continue
        
        if encodings:
            # Convert to numpy array and add to FAISS index
            encodings_array = np.array(encodings).astype('float32')
            face_index.add(encodings_array)
            logger.info(f"Updated FAISS index with {len(encodings)} face encodings")
            
            # Save the updated index
            save_faiss_index()
        
        session.close()
    except Exception as e:
        logger.error(f"Error updating FAISS index: {str(e)}")
        logger.error(traceback.format_exc())

def check_duplicate_face(encoding):
    """Check if a face encoding already exists in the database"""
    if face_index is not None and face_index.ntotal > 0:
        query = encoding.astype('float32').reshape(1, -1)
        distances, _ = face_index.search(query, 1)
        return distances[0][0] < FACE_DISTANCE_THRESHOLD
    return False

def check_duplicate_email(email):
    """Check if an email already exists in the database"""
    session = Session()
    try:
        existing_user = session.query(User).filter(User.email == email).first()
        return existing_user is not None
    finally:
        session.close()

# Initialize FAISS index on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing FAISS index...")
    if not load_faiss_index():
        initialize_faiss()
        update_faiss_index()
    logger.info("Waiting for camera warm-up...")
    time.sleep(5)

@app.on_event("shutdown")
async def shutdown_event():
    stop_event.set()
    save_faiss_index()

# ─── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    def stream():
        while True:
            try:
                frame = get_latest_frame()
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error in video feed: {str(e)}")
                time.sleep(1)  # Wait a bit before retrying
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

# Pydantic models for request/response
class UserCreate(BaseModel):
    name: str
    email: str

class BatchRegistrationItem(BaseModel):
    name: str
    email: str
    photo_path: str

@app.post("/register")
async def register_user(name: str = Form(...), email: str = Form(...), photo: UploadFile = File(...)):
    """Register a new user with their photo and face encoding"""
    try:
        # Check for duplicate email
        if check_duplicate_email(email):
            raise HTTPException(status_code=400, detail="This email is already registered.")

        # Read the uploaded image
        try:
            contents = await photo.read()
            if not contents:
                logger.error("Empty file uploaded")
                raise HTTPException(status_code=400, detail="Empty file uploaded")
                
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None: 
                logger.error("Could not decode uploaded image")
                raise HTTPException(status_code=400, detail="Could not decode uploaded image. Please ensure the file is a valid image.")

            # Log image properties for debugging
            logger.info(f"Image shape: {img.shape}, dtype: {img.dtype}")
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

        # Convert image to RGB for face_recognition
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error converting image to RGB: {str(e)}")
            raise HTTPException(status_code=400, detail="Error processing image color format")

        # Get face encoding from the uploaded photo
        try:
            face_locations = face_recognition.face_locations(rgb_img)
            if not face_locations:
                logger.warning("No face detected in image")
                raise HTTPException(status_code=400, detail="No face detected in the uploaded photo. Please ensure the photo contains a clear, front-facing image of a face.")
                
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            if not face_encodings:
                logger.error("Could not compute face encoding")
                raise HTTPException(status_code=500, detail="Could not process face in the uploaded photo. Please try a different photo.")

            encoding = face_encodings[0]
            encoding_bytes = encoding.tobytes()
        except Exception as e:
            logger.error(f"Error in face detection/encoding: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing face in the photo")

        # Check for duplicate face
        if check_duplicate_face(encoding):
            raise HTTPException(status_code=400, detail="This face is already registered.")

        # Resize photo for history view
        try:
            max_size = (150, 150)
            img_small = img.copy()
            h, w = img_small.shape[:2]
            if h > max_size[1] or w > max_size[0]:
                scale = max_size[0] / w if w > h else max_size[1] / h
                img_small = cv2.resize(img_small, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            _, registered_photo_small_buffer = cv2.imencode('.jpg', img_small)
            registered_photo_small_bytes = registered_photo_small_buffer.tobytes()
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing image for storage")

        # Create new user
        session = Session()
        try:
            new_user = User(
                name=name,
                email=email,
                face_encoding=encoding_bytes,
                registered_photo_small=registered_photo_small_bytes
            )

            session.add(new_user)
            session.commit()

            # Update FAISS index
            try:
                if face_index is not None:
                    encoding_np = np.frombuffer(encoding_bytes, dtype=np.float32).reshape(1, -1)
                    face_index.add(encoding_np)
                    user_mapping[face_index.ntotal - 1] = new_user.id
                    save_faiss_index()
                    logger.info("FAISS index updated and saved.")
            except Exception as faiss_error:
                logger.error(f"Error updating or saving FAISS index for user {new_user.email}: {str(faiss_error)}")
                # This is a non-critical error for the user registration itself, as data is in DB
                # However, the FAISS index will be out of sync. Log and continue.
                # We don't re-raise as a critical error here to allow the user registration to succeed.

            logger.info(f"User '{name}' registered successfully.")
            return JSONResponse(status_code=200, content={"message": f"Successfully registered user: {name}"})
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error saving user data to database")
        finally:
            session.close()

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in register_user: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

@app.post("/register/batch")
async def batch_register(users: list[BatchRegistrationItem]):
    """Register multiple users from a list of image paths"""
    results = []
    session = Session()
    
    try:
        for user in users:
            try:
                # Check for duplicate email
                if check_duplicate_email(user.email):
                    results.append({
                        "email": user.email,
                        "status": "failed",
                        "message": "Email already registered"
                    })
                    continue

                # Read and process image
                img = cv2.imread(user.photo_path)
                if img is None:
                    results.append({
                        "email": user.email,
                        "status": "failed",
                        "message": "Could not read image file"
                    })
                    continue

                # Convert image to RGB for face_recognition
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get face encoding
                face_locations = face_recognition.face_locations(rgb_img)
                if not face_locations:
                    results.append({
                        "email": user.email,
                        "status": "failed",
                        "message": "No face detected in image"
                    })
                    continue

                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                if not face_encodings:
                    results.append({
                        "email": user.email,
                        "status": "failed",
                        "message": "Could not process face"
                    })
                    continue

                encoding = face_encodings[0]
                encoding_bytes = encoding.tobytes()

                # Check for duplicate face
                if check_duplicate_face(encoding):
                    results.append({
                        "email": user.email,
                        "status": "failed",
                        "message": "Face already registered"
                    })
                    continue

                # Resize photo for history view
                max_size = (150, 150)
                img_small = img.copy()
                h, w = img_small.shape[:2]
                if h > max_size[1] or w > max_size[0]:
                    scale = max_size[0] / w if w > h else max_size[1] / h
                    img_small = cv2.resize(img_small, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                _, registered_photo_small_buffer = cv2.imencode('.jpg', img_small)
                registered_photo_small_bytes = registered_photo_small_buffer.tobytes()

                # Create new user
                new_user = User(
                    name=user.name,
                    email=user.email,
                    face_encoding=encoding_bytes,
                    registered_photo_small=registered_photo_small_bytes
                )

                session.add(new_user)
                session.flush()  # Get the ID without committing

                # Update FAISS index
                if face_index is not None:
                    encoding_np = np.frombuffer(encoding_bytes, dtype=np.float32).reshape(1, -1)
                    face_index.add(encoding_np)
                    user_mapping[face_index.ntotal - 1] = new_user.id

                results.append({
                    "email": user.email,
                    "status": "success",
                    "message": "User registered successfully"
                })

            except Exception as e:
                logger.error(f"Error processing user {user.email}: {str(e)}")
                results.append({
                    "email": user.email,
                    "status": "failed",
                    "message": str(e)
                })

        # Commit all successful registrations
        try:
            session.commit()
            logger.info("Batch registration database commit successful")
        except Exception as db_error:
            session.rollback()
            logger.error(f"Database commit error during batch registration: {str(db_error)}")
            logger.error(traceback.format_exc())
            # Update results for users that might have succeeded before the commit failed
            for res in results:
                if res["status"] == "success":
                     res["status"] = "failed"
                     res["message"] = f"Database commit failed: {str(db_error)}"

            raise HTTPException(status_code=500, detail=f"Database error during batch commit: {str(db_error)}") # Re-raise to be caught by outer handler

        # Save FAISS index after batch registration
        # This will only happen if the database commit was successful
        try:
            save_faiss_index()
            logger.info("FAISS index saved after batch registration")
        except Exception as faiss_error:
             logger.error(f"Error saving FAISS index after batch registration: {str(faiss_error)}")
             # This is a warning, not a critical failure for the user registration itself
             # The user data is in the DB, but FAISS might be out of sync until next startup/update


        return JSONResponse(status_code=200, content={
            "message": "Batch registration completed",
            "results": results
        })

    except HTTPException as e:
        raise e # Re-raise HTTPException to be handled by FastAPI
    except Exception as e:
        session.rollback() # Ensure rollback on any other unexpected error
        logger.error(f"Unexpected error in batch registration: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error during batch registration")
    finally:
        session.close()

@app.get("/authenticate")
async def authenticate():
    """Authenticate user from live camera feed using FAISS for fast face matching"""
    try:
        # Wait for frames to be available
        max_retries = 3
        retry_count = 0
        
        while frame_queue.empty() and retry_count < max_retries:
            logger.info(f"Waiting for frames... Attempt {retry_count + 1}/{max_retries}")
            time.sleep(1)
            retry_count += 1
            
        if frame_queue.empty():
            logger.warning("No frames available for authentication after retries")
            return JSONResponse(status_code=200, content={"status": "No frames available", "message": "Please try again"})
            
        # Get the frame from the queue
        frame = frame_queue.get() 
        if frame is None:
            logger.error("Failed to get frame from queue")
            return JSONResponse(status_code=500, content={"status": "Error", "message": "Could not read from camera"})
        
        # Convert frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get face locations and encodings from the live frame
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            logger.info("No face detected in frame")
            return JSONResponse(status_code=200, content={"status": "No face detected"})
            
        live_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not live_face_encodings:
            logger.error("Could not compute encoding for live face")
            return JSONResponse(status_code=500, content={"status": "Error", "message": "Could not process face in live feed."})

        # Use the first detected face encoding
        live_encoding = live_face_encodings[0]
        
        # Convert to float32 for FAISS
        live_encoding = live_encoding.astype('float32').reshape(1, -1)
        
        # Search in FAISS index
        if face_index is not None and face_index.ntotal > 0:
            # Search for the closest match
            distances, indices = face_index.search(live_encoding, 1)
            
            # Check if the match is within threshold
            if distances[0][0] < FACE_DISTANCE_THRESHOLD:  # Lower distance means better match
                matched_user_id = user_mapping[indices[0][0]]
                
                # Get user details from database
                session = Session()
                user = session.query(User).filter(User.id == matched_user_id).first()
                
                if user:
                    # Encode the live frame to JPEG bytes for storing in history
                    _, live_capture_buffer = cv2.imencode('.jpg', frame)
                    live_capture_bytes = live_capture_buffer.tobytes()

                    login_record = LoginHistory(
                        user_id=user.id,
                        name=user.name,
                        registered_image=user.registered_photo_small,
                        live_capture=live_capture_bytes,
                        status="Granted"
                    )
                    session.add(login_record)
                    session.commit()
                    session.close()
                    
                    logger.info(f"Authentication successful for user: {user.name}")
                    return JSONResponse(status_code=200, content={"status": "Access Granted", "user": user.name})
                session.close()
        
        # If no match found or match is not good enough
        # Encode the live frame to JPEG bytes for storing in history
        _, live_capture_buffer = cv2.imencode('.jpg', frame)
        live_capture_bytes = live_capture_buffer.tobytes()

        login_record = LoginHistory(
            user_id=0,
            name="Unknown",
            registered_image=b"",
            live_capture=live_capture_bytes,
            status="Denied"
        )
        session = Session()
        session.add(login_record)
        session.commit()
        session.close()
        
        logger.info("Authentication failed - unauthorized")
        return JSONResponse(status_code=200, content={"status": "Access Denied - Unauthorized"})
        
    except Exception as e:
        logger.error(f"Error in authenticate endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"status": "Error", "message": "An internal server error occurred."})

@app.get("/api/history")
async def get_login_history():
    """Get login history"""
    try:
        session = Session()
        records = session.query(LoginHistory).order_by(LoginHistory.login_time.desc()).all()
        
        response = [record.to_dict() for record in records]
        session.close()
        return response
    except Exception as e:
        logger.error(f"Error in get_login_history: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_system_metrics():
    """Get system metrics including user count, recent logins, and FAISS index size"""
    try:
        session = Session()
        
        # Get total user count
        user_count = session.query(User).count()
        
        # Get recent login statistics (last 24 hours)
        recent_time = datetime.utcnow() - timedelta(hours=24)
        recent_logins = session.query(LoginHistory).filter(
            LoginHistory.login_time >= recent_time
        ).all()
        
        granted_count = sum(1 for login in recent_logins if login.status == "Granted")
        denied_count = sum(1 for login in recent_logins if login.status == "Denied")
        
        # Get FAISS index size
        faiss_size = face_index.ntotal if face_index is not None else 0
        
        # Calculate system uptime
        uptime = datetime.utcnow() - STARTUP_TIME
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        session.close()
        
        return {
            "user_count": user_count,
            "recent_logins": {
                "granted": granted_count,
                "denied": denied_count,
                "total": len(recent_logins)
            },
            "faiss_index_size": faiss_size,
            "system_uptime": uptime_str
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ─── Run App ──────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
