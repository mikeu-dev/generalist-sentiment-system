import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)) # 16MB
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

class ProductionConfig(Config):
    """Production specific config."""
    DEBUG = False

class DevelopmentConfig(Config):
    """Development specific config."""
    DEBUG = True
