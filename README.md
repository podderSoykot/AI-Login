# AI Login

A face recognition-based authentication system with multi-camera support.

## Features

- Face recognition using FAISS for fast matching
- Multi-camera support with RTSP streams
- Camera location tracking
- Batch user registration
- Duplicate face and email detection
- Login history with camera information
- Persistent storage of face encodings

## Requirements

- Python 3.8+
- OpenCV
- face_recognition
- FAISS
- FastAPI
- SQLAlchemy
- uvicorn

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd face-auth-system
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create required directories:

```bash
mkdir static
```

## Usage

1. Start the server:

```bash
python main.py
```

2. Access the web interface at `http://localhost:8000`

## API Endpoints

- `GET /` - Home page
- `GET /register` - Registration page
- `GET /login` - Login page
- `GET /history` - Login history page
- `GET /cameras` - Camera management page
- `POST /register` - Register a single user
- `POST /register/batch` - Register multiple users
- `GET /authenticate` - Authenticate user from camera feed
- `GET /api/history` - Get login history
- `GET /api/cameras` - Get camera list
- `POST /api/cameras` - Add a new camera
- `DELETE /api/cameras/{camera_id}` - Delete a camera

## Version History

### Version 1

- Initial release with core functionality
- FAISS persistence
- Multi-camera support
- Batch registration
- Camera location tracking
