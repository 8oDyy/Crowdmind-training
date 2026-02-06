"""HTTP client for CrowdMind API communication.

This module handles all HTTP interactions with the backend API:
- Downloading datasets (CSV + schema.json)
- Uploading trained models (model.tflite + meta.json)

No API logic is implemented here - only HTTP calls.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class CrowdMindAPIClient:
    """HTTP client for CrowdMind API.
    
    Attributes:
        base_url: Base URL of the CrowdMind API.
        token: Authentication token for API requests.
        timeout: Request timeout in seconds.
    """
    
    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: int = 30
    ) -> None:
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the CrowdMind API (e.g., "https://api.crowdmind.com").
            token: Bearer token for authentication.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
    
    def _get_headers(self, content_type: Optional[str] = None) -> dict:
        """Get request headers with optional content type override."""
        headers = {"Authorization": f"Bearer {self.token}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers
    
    def download_dataset(
        self,
        dataset_id: str,
        output_dir: Path
    ) -> tuple[Path, Path]:
        """Download dataset CSV and schema from the API.
        
        Args:
            dataset_id: UUID of the dataset to download.
            output_dir: Directory to save downloaded files.
        
        Returns:
            Tuple of (csv_path, schema_path).
        
        Raises:
            requests.HTTPError: If the API request fails.
            ValueError: If the response is invalid.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_url = f"{self.base_url}/datasets/{dataset_id}/download"
        logger.info(f"Downloading dataset CSV from: {csv_url}")
        
        response = self._session.get(csv_url, timeout=self.timeout)
        response.raise_for_status()
        
        csv_path = output_dir / "dataset.csv"
        csv_path.write_bytes(response.content)
        logger.info(f"Dataset CSV saved to: {csv_path}")
        
        schema_url = f"{self.base_url}/datasets/{dataset_id}/schema"
        logger.info(f"Downloading schema from: {schema_url}")
        
        response = self._session.get(schema_url, timeout=self.timeout)
        response.raise_for_status()
        
        schema_path = output_dir / "schema.json"
        schema_data = response.json()
        schema_path.write_text(json.dumps(schema_data, indent=2))
        logger.info(f"Schema saved to: {schema_path}")
        
        return csv_path, schema_path
    
    def upload_model(
        self,
        dataset_id: str,
        model_path: Path,
        meta_path: Path
    ) -> dict:
        """Upload trained model and metadata to the API.
        
        Args:
            dataset_id: UUID of the dataset used for training.
            model_path: Path to the model.tflite file.
            meta_path: Path to the meta.json file.
        
        Returns:
            API response as dictionary.
        
        Raises:
            requests.HTTPError: If the API request fails.
            FileNotFoundError: If model or meta files don't exist.
        """
        model_path = Path(model_path)
        meta_path = Path(meta_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        
        upload_url = f"{self.base_url}/datasets/{dataset_id}/models"
        logger.info(f"Uploading model to: {upload_url}")
        
        with open(model_path, "rb") as model_file, open(meta_path, "r") as meta_file:
            files = {
                "model": ("model.tflite", model_file, "application/octet-stream"),
                "meta": ("meta.json", meta_file, "application/json"),
            }
            headers = {"Authorization": f"Bearer {self.token}"}
            
            response = requests.post(
                upload_url,
                files=files,
                headers=headers,
                timeout=self.timeout
            )
        
        response.raise_for_status()
        result = response.json()
        logger.info(f"Model uploaded successfully. Response: {result}")
        
        return result
    
    def health_check(self) -> bool:
        """Check if the API is reachable.
        
        Returns:
            True if API is healthy, False otherwise.
        """
        try:
            response = self._session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Health check failed: {e}")
            return False
