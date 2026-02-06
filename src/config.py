"""
App configuration loader.
- Credentials: loaded from .env
- App config: loaded from config.json (can be reloaded at runtime)
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
class EliteScaling:
    threshold: int = 500000
    base_offset: float = 0.25
    scaling_factor: float = 0.5


@dataclass
class ConfidenceTiers:
    close_threshold: float = 0.7
    extreme_threshold: float = 1.0


@dataclass
class RegionConfig:
    model: str
    currency_id: int
    hist_countries: list[str]
    elite_scaling: EliteScaling
    confidence_tiers: ConfidenceTiers
    sire_sample_min_count: int = 10


@dataclass
class AppConfig:
    year_start: int
    year_end: int | None
    model_test_last_years: int
    sale_history_years: int
    audit_user_id: int
    regions: dict[str, RegionConfig]

    def get_year_end(self) -> int:
        """Get year_end - uses current year if None."""
        from datetime import datetime
        return self.year_end if self.year_end is not None else datetime.now().year

    def get_region(self, country: str) -> RegionConfig:
        """Get region config for a country."""
        country = country.upper()
        if country not in self.regions:
            raise ValueError(f"Unknown region: {country}")
        return self.regions[country]

    def get_model_dir(self, country: str) -> str:
        """Get model directory for a country."""
        region = self.get_region(country)
        return f"models/{region.model}"

    def get_hist_countries(self, country: str) -> list[str]:
        """Get list of historical countries for lookback."""
        return self.get_region(country).hist_countries

    def get_currency_id(self, country: str) -> int:
        """Get currency ID for a country."""
        return self.get_region(country).currency_id

    def get_elite_scaling(self, country: str) -> EliteScaling:
        """Get elite scaling config for a country."""
        return self.get_region(country).elite_scaling

    def get_confidence_tiers(self, country: str) -> ConfidenceTiers:
        """Get confidence tier thresholds for a country."""
        return self.get_region(country).confidence_tiers

    def get_sire_sample_min_count(self, country: str) -> int:
        """Get minimum sire sample count for a country."""
        return self.get_region(country).sire_sample_min_count

    # Legacy accessors for backward compatibility
    @property
    def models(self) -> dict[str, str]:
        """Legacy accessor - returns {country: model} mapping."""
        return {k: v.model for k, v in self.regions.items()}

    @property
    def currency_map(self) -> dict[str, int]:
        """Legacy accessor - returns {country: currency_id} mapping."""
        return {k: v.currency_id for k, v in self.regions.items()}

    @property
    def hist_countries(self) -> dict[str, list[str]]:
        """Legacy accessor - returns {country: hist_countries} mapping."""
        return {k: v.hist_countries for k, v in self.regions.items()}


def _parse_region_config(data: dict) -> RegionConfig:
    """Parse region config from JSON dict."""
    elite_data = data.get("elite_scaling", {})
    confidence_data = data.get("confidence_tiers", {})

    return RegionConfig(
        model=data["model"],
        currency_id=data["currency_id"],
        hist_countries=data["hist_countries"],
        elite_scaling=EliteScaling(
            threshold=elite_data.get("threshold", 500000),
            base_offset=elite_data.get("base_offset", 0.25),
            scaling_factor=elite_data.get("scaling_factor", 0.5),
        ),
        confidence_tiers=ConfidenceTiers(
            close_threshold=confidence_data.get("close_threshold", 0.7),
            extreme_threshold=confidence_data.get("extreme_threshold", 1.0),
        ),
        sire_sample_min_count=data.get("sire_sample_min_count", 10),
    )


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

        regions = {
            k: _parse_region_config(v)
            for k, v in data.get("regions", {}).items()
        }

        return AppConfig(
            year_start=data["year_start"],
            year_end=data.get("year_end"),
            model_test_last_years=data.get("model_test_last_years", 2),
            sale_history_years=data.get("sale_history_years", 5),
            audit_user_id=data.get("audit_user_id", 2),
            regions=regions,
        )

    def reload(self):
        """Reload app config from JSON file."""
        self.app = self._load_app_config()


# Singleton
config = Config()
