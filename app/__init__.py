"""Waste Scanner AI package configuration."""

from pathlib import Path

from dotenv import load_dotenv

# Always load the project's .env before submodules read environment variables.
# Existing OS/container variables still take precedence because override=False.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
