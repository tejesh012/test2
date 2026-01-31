from flask import Flask
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_bcrypt import Bcrypt
from app.config import config

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
bcrypt = Bcrypt()


def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    bcrypt.init_app(app)

    # Configure CORS - Allow specific origins + any Vercel preview URLs
    frontend_url = os.getenv('FRONTEND_URL')

    # Base allowed origins
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        r"^https://.*\.vercel\.app$"  # Regex to match any vercel.app subdomain
    ]

    if frontend_url:
        allowed_origins.append(frontend_url)

    CORS(app, resources={
        r"/api/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.user import user_bp
    from app.routes.face_detection import face_detection_bp
    from app.routes.chat import chat_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(user_bp, url_prefix='/api/user')
    app.register_blueprint(face_detection_bp, url_prefix='/api/detection')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')

    # Health check
    @app.route('/api/health')
    def health_check():
        return {'status': 'healthy', 'message': 'AI Demo API is running'}

    return app
