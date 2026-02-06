"""HTTP client for CrowdMind API communication.

This module handles all HTTP interactions with the backend API:
- Managing datasets and dataset versions
- Managing models and model versions
- Downloading/uploading files via Supabase Storage

API structure:
- Datasets: POST /datasets, POST /datasets/{id}/versions, POST /datasets/{id}/generate
- Models: POST /models, POST /models/{id}/versions
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
        token: str = "",
        timeout: int = 30
    ) -> None:
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the CrowdMind API (e.g., "https://api.crowdmind.fr/api/v1").
            token: Bearer token for authentication (optional for now).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._session.headers.update(headers)
    
    # =========================================================================
    # HEALTH
    # =========================================================================
    
    def health_check(self) -> bool:
        """Check if the API is reachable."""
        try:
            response = self._session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    # =========================================================================
    # DATASETS
    # =========================================================================
    
    def create_dataset(
        self,
        name: str,
        dataset_type: str,
        created_by: str,
        description: Optional[str] = None
    ) -> dict:
        """Create a new dataset.
        
        Args:
            name: Dataset name.
            dataset_type: Type of dataset (synthetic, scraped, mixed).
            created_by: User ID who created the dataset.
            description: Optional description.
        
        Returns:
            API response with dataset info including ID.
        """
        url = f"{self.base_url}/datasets"
        logger.info(f"Creating dataset: {name}")
        
        payload = {
            "name": name,
            "dataset_type": dataset_type,
            "created_by": created_by,
        }
        if description:
            payload["description"] = description
        
        response = self._session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Dataset created: {result['id']}")
        return result
    
    def get_dataset(self, dataset_id: str) -> dict:
        """Get dataset info by ID."""
        url = f"{self.base_url}/datasets/{dataset_id}"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """List all datasets."""
        url = f"{self.base_url}/datasets"
        params = {"limit": limit, "offset": offset}
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def create_dataset_version(
        self,
        dataset_id: str,
        version: str,
        file_path: Path,
        format: str = "csv"
    ) -> dict:
        """Upload a new version of a dataset.
        
        Args:
            dataset_id: UUID of the dataset.
            version: Version string (e.g., "1.0", "2.0").
            file_path: Path to the data file to upload.
            format: File format (csv, jsonl, parquet).
        
        Returns:
            API response with version info.
        """
        url = f"{self.base_url}/datasets/{dataset_id}/versions"
        params = {"version": version, "format": format}
        
        logger.info(f"Uploading dataset version {version} for {dataset_id}")
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            
            response = requests.post(
                url,
                params=params,
                files=files,
                headers=headers,
                timeout=self.timeout
            )
        
        response.raise_for_status()
        result = response.json()
        logger.info(f"Dataset version uploaded: {result['id']}")
        return result
    
    def generate_synthetic_dataset(
        self,
        dataset_id: str,
        version: str,
        n: int = 100,
        seed: Optional[int] = None,
        labels: Optional[list[str]] = None
    ) -> dict:
        """Generate a synthetic dataset version.
        
        Args:
            dataset_id: UUID of the dataset.
            version: Version string for the generated data.
            n: Number of rows to generate (1-10000).
            seed: Optional random seed for reproducibility.
            labels: Optional list of class labels.
        
        Returns:
            API response with version info.
        """
        url = f"{self.base_url}/datasets/{dataset_id}/generate"
        
        payload = {"version": version, "n": n}
        if seed is not None:
            payload["seed"] = seed
        if labels:
            payload["labels"] = labels
        
        logger.info(f"Generating {n} synthetic rows for dataset {dataset_id}")
        
        response = self._session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Synthetic version created: {result['id']}")
        return result
    
    def list_dataset_versions(self, dataset_id: str) -> list[dict]:
        """List all versions of a dataset."""
        url = f"{self.base_url}/datasets/{dataset_id}/versions"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_dataset_version_download_url(
        self,
        version_id: str,
        expires: int = 3600
    ) -> str:
        """Get a signed download URL for a dataset version.
        
        Args:
            version_id: UUID of the dataset version.
            expires: URL expiration time in seconds (60-86400).
        
        Returns:
            Signed download URL.
        """
        url = f"{self.base_url}/datasets/versions/{version_id}/download"
        params = {"expires": expires}
        
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["url"]
    
    def download_dataset_version(
        self,
        version_id: str,
        output_dir: Path
    ) -> Path:
        """Download a dataset version file.
        
        Args:
            version_id: UUID of the dataset version.
            output_dir: Directory to save the file.
        
        Returns:
            Path to the downloaded file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        download_url = self.get_dataset_version_download_url(version_id)
        logger.info(f"Downloading dataset version {version_id}")
        
        response = requests.get(download_url, timeout=self.timeout)
        response.raise_for_status()
        
        file_path = output_dir / "dataset.csv"
        file_path.write_bytes(response.content)
        logger.info(f"Dataset saved to: {file_path}")
        
        return file_path
    
    # =========================================================================
    # MODELS
    # =========================================================================
    
    def create_model(
        self,
        name: str,
        framework: str,
        description: Optional[str] = None
    ) -> dict:
        """Create a new model.
        
        Args:
            name: Model name.
            framework: ML framework (edge_impulse, tflite, custom_mlp).
            description: Optional description.
        
        Returns:
            API response with model info including ID.
        """
        url = f"{self.base_url}/models"
        logger.info(f"Creating model: {name}")
        
        payload = {
            "name": name,
            "framework": framework,
        }
        if description:
            payload["description"] = description
        
        response = self._session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Model created: {result['id']}")
        return result
    
    def get_model(self, model_id: str) -> dict:
        """Get model info by ID."""
        url = f"{self.base_url}/models/{model_id}"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def list_models(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """List all models."""
        url = f"{self.base_url}/models"
        params = {"limit": limit, "offset": offset}
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def create_model_version(
        self,
        model_id: str,
        version: str,
        file_path: Path
    ) -> dict:
        """Upload a new version of a model.
        
        Args:
            model_id: UUID of the model.
            version: Version string (e.g., "1.0.0").
            file_path: Path to the model file (.tflite).
        
        Returns:
            API response with version info.
        """
        url = f"{self.base_url}/models/{model_id}/versions"
        params = {"version": version}
        
        logger.info(f"Uploading model version {version} for {model_id}")
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            
            response = requests.post(
                url,
                params=params,
                files=files,
                headers=headers,
                timeout=self.timeout
            )
        
        response.raise_for_status()
        result = response.json()
        logger.info(f"Model version uploaded: {result['id']}")
        return result
    
    def list_model_versions(self, model_id: str) -> list[dict]:
        """List all versions of a model."""
        url = f"{self.base_url}/models/{model_id}/versions"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_model_version_download_url(
        self,
        version_id: str,
        expires: int = 3600
    ) -> str:
        """Get a signed download URL for a model version."""
        url = f"{self.base_url}/models/versions/{version_id}/download"
        params = {"expires": expires}
        
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["url"]
