import os
import secrets
import logging

logger = logging.getLogger(__name__)

class Config:
    """Base configuration."""
    # Security: SECRET_KEY harus di-set via environment variable di production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    if not SECRET_KEY:
        if os.environ.get('FLASK_ENV') == 'production':
            raise ValueError(
                "SECRET_KEY must be set in production! "
                "Set environment variable: export SECRET_KEY='your-secret-key'"
            )
        else:
            # Generate random key untuk development
            SECRET_KEY = secrets.token_hex(32)
            logger.warning("Using generated SECRET_KEY for development. DO NOT use in production!")
    
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///sentiment.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Rate Limiting Configuration
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/1')
    RATELIMIT_STRATEGY = 'fixed-window'
    RATELIMIT_HEADERS_ENABLED = True
    
    # Security Headers
    SESSION_COOKIE_SECURE = os.environ.get('FLASK_ENV') == 'production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class ProductionConfig(Config):
    """Production specific config."""
    DEBUG = False
    TESTING = False
    
    # Enforce HTTPS in production
    SESSION_COOKIE_SECURE = True

class DevelopmentConfig(Config):
    """Development specific config."""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing specific config."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
