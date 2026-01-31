from app.routes.auth import auth_bp
from app.routes.user import user_bp
from app.routes.face_detection import face_detection_bp
from app.routes.chat import chat_bp

__all__ = ['auth_bp', 'user_bp', 'face_detection_bp', 'chat_bp']
