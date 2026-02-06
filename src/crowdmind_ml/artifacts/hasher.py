"""Dataset hashing for reproducibility tracking."""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetHasher:
    """Computes SHA256 hash of dataset files.
    
    Used to track dataset versions and ensure reproducibility.
    """
    
    @staticmethod
    def hash_file(file_path: Path) -> str:
        """Compute SHA256 hash of a file.
        
        Args:
            file_path: Path to the file to hash.
        
        Returns:
            Hexadecimal SHA256 hash string.
        
        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        hash_value = sha256_hash.hexdigest()
        logger.info(f"Hash of {file_path.name}: {hash_value[:16]}...")
        
        return hash_value
    
    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Compute SHA256 hash of raw bytes.
        
        Args:
            data: Bytes to hash.
        
        Returns:
            Hexadecimal SHA256 hash string.
        """
        return hashlib.sha256(data).hexdigest()
