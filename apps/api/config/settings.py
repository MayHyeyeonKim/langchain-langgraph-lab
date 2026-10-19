from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR.parent.parent


class DjangoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_DIR / ".env", extra="ignore")
    django_secret_key: str = "unsafe-development-key"
    django_debug: bool = True


env = DjangoSettings()
SECRET_KEY = env.django_secret_key
DEBUG = env.django_debug
ALLOWED_HOSTS: list[str] = []
ROOT_URLCONF = "config.urls"
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "rest_framework",
    "reviews",
]
MIDDLEWARE = []
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "UNAUTHENTICATED_USER": None,
}
