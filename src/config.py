"""
App configuration loader.
- Credentials: loaded from .env
- App config: loaded from config.json (can be reloaded at runtime)
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


@dataclass
class DatabaseCredentials:
    server: str
    name: str
    user: str
    password: str


@dataclass
class AppConfig:
    models: dict[str, str]
    year_start: int
    year_end: int | None
    model_test_last_years: int
    audit_user_id: int
    hist_countries: dict[str, list[str]]
    currency_map: dict[str, int]

    def get_year_end(self) -> int:
        """Get year_end - uses current year if None."""
        from datetime import datetime
        return self.year_end if self.year_end is not None else datetime.now().year

    def get_model_dir(self, country: str) -> str:
        """Get model directory for a country."""
        model = self.models.get(country.lower(), country.lower())
        return f"models/{model}"

    def get_hist_countries(self, country: str) -> list[str]:
        """Get list of historical countries for lookback."""
        return self.hist_countries.get(country.upper(), [country.upper()])


class Config:
    def __init__(self):
        self.db = self._load_db_credentials()
        self.app = self._load_app_config()

    def _load_db_credentials(self) -> DatabaseCredentials:
        return DatabaseCredentials(
            server=os.getenv("DB_SERVER"),
            name=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )

    def _load_app_config(self) -> AppConfig:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        return AppConfig(**data)

    def reload(self):
        """Reload app config from JSON file."""
        self.app = self._load_app_config()


# Singleton
config = Config()
