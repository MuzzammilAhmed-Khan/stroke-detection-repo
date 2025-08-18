"""
Configuration settings for the stroke prediction application
"""
import os
from datetime import timedelta


class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Model settings
    MODEL_PATH_PRIMARY = 'XGBstroke.pkl'
    MODEL_PATH_SECONDARY = 'XGB_RFstroke.pkl'
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    
    # Security settings
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False
    
    # Development-specific settings
    TEMPLATES_AUTO_RELOAD = True


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    
    # Use in-memory models for testing
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'stroke_prediction.log'
    
    # Performance settings
    SEND_FILE_MAX_AGE_DEFAULT = timedelta(days=365)


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}