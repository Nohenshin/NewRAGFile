# evaluation/ground_truth.py
import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .config import GROUND_TRUTH_DIR

logger = logging.getLogger(__name__)

class GroundTruthLoader:
    """
    Load và validate ground truth data từ file JSON.
    """

    def __init__(self, ground_truth_dir: Optional[Path] = None):
        self.ground_truth_dir = ground_truth_dir or GROUND_TRUTH_DIR
        os.makedirs(self.ground_truth_dir, exist_ok=True)
        logger.info(f"Ground truth dir: {self.ground_truth_dir}")

    def get_available_files(self) -> List[str]:
        """Liệt kê các file JSON có sẵn."""
        return [f.name for f in self.ground_truth_dir.glob("*.json")]

    def validate_file(self, filename: str) -> bool:
        """Kiểm tra file có đúng cấu trúc không."""
        file_path = self.ground_truth_dir / filename
        if not file_path.exists():
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "questions" not in data:
                return False
            if not data["questions"]:
                return False
            first = data["questions"][0]
            if "question" not in first or "answer" not in first:
                return False
            return True
        except:
            return False

    def load_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load toàn bộ file JSON."""
        if not self.validate_file(filename):
            return None
        file_path = self.ground_truth_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_questions(self, filename: str) -> Optional[List[Dict[str, str]]]:
        """Load danh sách câu hỏi - câu trả lời."""
        data = self.load_data(filename)
        if data is None:
            return None
        return data.get("questions", [])