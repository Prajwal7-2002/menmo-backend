import os
from datetime import timedelta
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


def env_list(name: str) -> list:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


# -------------------------------------------
# RUNTIME MODE
# -------------------------------------------
# Hugging Face Spaces set SPACE_ID; a configured Postgres host also means prod.
ON_HF_SPACE = bool(os.getenv("SPACE_ID"))
USE_POSTGRES = os.getenv("DB_HOST") not in (None, "", "None")
IS_LOCAL = not ON_HF_SPACE and not USE_POSTGRES

# -----------------------------
# SECURITY
# -----------------------------
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if not IS_LOCAL:
        # A default key in a public repo would let anyone forge login tokens.
        raise ImproperlyConfigured("SECRET_KEY must be set in production (add it as a Space secret).")
    SECRET_KEY = "local-dev-only-secret-key"

DEBUG = env_bool("DJANGO_DEBUG", IS_LOCAL)

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS")
if not ALLOWED_HOSTS:
    if IS_LOCAL:
        ALLOWED_HOSTS = ["*"]
    else:
        ALLOWED_HOSTS = [h for h in (os.getenv("SPACE_HOST"), ".hf.space", "localhost", "127.0.0.1") if h]

# HF terminates TLS at its proxy
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# -----------------------------
# APPS
# -----------------------------
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'rest_framework',
    'rest_framework_simplejwt',
    "corsheaders",

    'auth_app',
    'api_app',
    'rag',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    # Limits LLM spend and brute-force attempts; tune via env.
    'DEFAULT_THROTTLE_CLASSES': ('rest_framework.throttling.ScopedRateThrottle',),
    'DEFAULT_THROTTLE_RATES': {
        'ask': os.getenv("THROTTLE_ASK", "30/min"),
        'upload': os.getenv("THROTTLE_UPLOAD", "20/hour"),
        'auth': os.getenv("THROTTLE_AUTH", "10/min"),
    },
}

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
    'AUTH_HEADER_TYPES': ('Bearer',),
}

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'copilot_backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'copilot_backend.wsgi.application'

# -------------------------------------------
# DATABASE
# -------------------------------------------
if USE_POSTGRES:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("DB_NAME"),
            "USER": os.getenv("DB_USER"),
            "PASSWORD": os.getenv("DB_PASSWORD"),
            "HOST": os.getenv("DB_HOST"),
            "PORT": os.getenv("DB_PORT", "5432"),
            "CONN_MAX_AGE": int(os.getenv("DB_CONN_MAX_AGE", "0")),
            "CONN_HEALTH_CHECKS": True,
            "OPTIONS": {
                "sslmode": os.getenv("DB_SSLMODE", "require"),
                "connect_timeout": 5,
                "application_name": "menmo_backend",
            },
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {"NAME": 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {"NAME": 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {"NAME": 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, "static")

# Uploads larger than this are streamed to a temp file instead of RAM.
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024

# -----------------------------
# CORS
# -----------------------------
# Auth is a Bearer token (not cookies), so open CORS is not a CSRF risk, but
# set CORS_ALLOWED_ORIGINS (comma separated) to lock it to your frontend.
CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS")
CORS_ALLOW_ALL_ORIGINS = not CORS_ALLOWED_ORIGINS

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# -----------------------------
# LOGGING (stdout -> Space logs)
# -----------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
    "loggers": {
        "django": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "httpx": {"level": "WARNING"},
        "sentence_transformers": {"level": "WARNING"},
    },
}
