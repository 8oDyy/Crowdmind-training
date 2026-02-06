"""Training run folder management."""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RunManager:
    """Manages training run folders with timestamps.
    
    Creates structured run folders containing:
    - Downloaded dataset
    - Model checkpoints
    - Final model.tflite
    - meta.json
    - Training logs
    """
    
    def __init__(self, runs_dir: Path) -> None:
        """Initialize the run manager.
        
        Args:
            runs_dir: Base directory for all training runs.
        """
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def create_run(self, dataset_id: str) -> Path:
        """Create a new timestamped run folder.
        
        Args:
            dataset_id: Dataset UUID for folder naming.
        
        Returns:
            Path to the created run folder.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = dataset_id[:8] if len(dataset_id) >= 8 else dataset_id
        run_name = f"run_{timestamp}_{short_id}"
        
        run_dir = self.runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        (run_dir / "data").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "output").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        
        logger.info(f"Created run folder: {run_dir}")
        
        return run_dir
    
    def get_data_dir(self, run_dir: Path) -> Path:
        """Get the data subdirectory for a run."""
        return run_dir / "data"
    
    def get_checkpoints_dir(self, run_dir: Path) -> Path:
        """Get the checkpoints subdirectory for a run."""
        return run_dir / "checkpoints"
    
    def get_output_dir(self, run_dir: Path) -> Path:
        """Get the output subdirectory for a run."""
        return run_dir / "output"
    
    def get_logs_dir(self, run_dir: Path) -> Path:
        """Get the logs subdirectory for a run."""
        return run_dir / "logs"
    
    def cleanup_run(self, run_dir: Path) -> None:
        """Remove a run folder and all its contents.
        
        Args:
            run_dir: Path to the run folder to remove.
        """
        if run_dir.exists():
            shutil.rmtree(run_dir)
            logger.info(f"Cleaned up run folder: {run_dir}")
    
    def list_runs(self) -> list[Path]:
        """List all existing run folders.
        
        Returns:
            List of run folder paths, sorted by name (newest first).
        """
        runs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        return sorted(runs, reverse=True)
