"""
Data Processor for TriMed-Agent Training
========================================

Converts raw medical data (VinDr-CXR, RAG Q&A) into training format.

Input Formats Supported:
1. VinDr-CXR style CSV with bounding boxes
2. RAG Q&A pairs in JSON
3. Custom instruction-planning pairs

Output Format:
{
    "image": "path/to/image.jpg",
    "conversations": [
        {"role": "user", "content": "<image>\nQuestion"},
        {"role": "assistant", "content": "{JSON plan}"}
    ]
}
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing"""
    raw_data_path: str = "data/raw"
    output_path: str = "data/sft_dataset"
    train_split: float = 0.9
    seed: int = 42
    
    # VinDr-CXR specific
    image_dir: str = "data/images"
    annotations_file: str = "data/annotations.csv"
    
    # RAG specific  
    rag_qa_file: str = "data/rag_qa.json"


class MedicalDataProcessor:
    """
    Process raw medical data into training format.
    
    Supports:
    1. VinDr-CXR: CSV with image_id, class_name, x_min, y_min, x_max, y_max
    2. RAG Q&A: JSON with question-answer pairs
    3. Custom: Pre-formatted instruction-planning pairs
    """
    
    # Tool mapping for action generation
    TOOL_MAPPING = {
        "nodule": {"action": "GroundingDINO", "prompt": "lung nodule"},
        "cardiomegaly": {"action": "GroundingDINO", "prompt": "enlarged heart"},
        "infiltrate": {"action": "GroundingDINO", "prompt": "lung infiltrate"},
        "effusion": {"action": "GroundingDINO", "prompt": "pleural effusion"},
        "mass": {"action": "GroundingDINO", "prompt": "lung mass"},
        "pneumonia": {"action": "GroundingDINO", "prompt": "pneumonia consolidation"},
        "atelectasis": {"action": "GroundingDINO", "prompt": "atelectasis"},
        "pneumothorax": {"action": "GroundingDINO", "prompt": "pneumothorax"},
        "fracture": {"action": "GroundingDINO", "prompt": "rib fracture"},
    }
    
    # Question templates for generating diverse training data
    QUESTION_TEMPLATES = [
        "Có bất thường gì trong ảnh này không?",
        "Hãy phân tích ảnh X-quang này.",
        "Tìm {target} trong ảnh.",
        "Ảnh này có {target} không?",
        "Đánh giá {target} trong ảnh X-quang.",
        "What abnormalities do you see in this image?",
        "Analyze this chest X-ray for {target}.",
        "Is there any {target} visible?",
    ]
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        random.seed(self.config.seed)
        
    def process_vindr_cxr(
        self,
        csv_path: str,
        image_dir: str,
        output_path: str
    ) -> Tuple[int, int]:
        """
        Process VinDr-CXR format CSV into training data.
        
        CSV columns: image_id, class_name, x_min, y_min, x_max, y_max
        
        Returns:
            (train_count, val_count)
        """
        logger.info(f"Processing VinDr-CXR data from {csv_path}")
        
        # Read CSV
        data_by_image = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['image_id']
                if image_id not in data_by_image:
                    data_by_image[image_id] = []
                    
                # Parse bbox
                bbox = [
                    float(row.get('x_min', 0)),
                    float(row.get('y_min', 0)),
                    float(row.get('x_max', 0)),
                    float(row.get('y_max', 0))
                ]
                
                data_by_image[image_id].append({
                    'class_name': row.get('class_name', 'abnormality'),
                    'bbox': bbox
                })
                
        logger.info(f"Found {len(data_by_image)} unique images")
        
        # Convert to training format
        examples = []
        for image_id, annotations in data_by_image.items():
            image_path = Path(image_dir) / f"{image_id}.jpg"
            if not image_path.exists():
                image_path = Path(image_dir) / f"{image_id}.png"
                
            for ann in annotations:
                example = self._create_detection_example(
                    str(image_path),
                    ann['class_name'],
                    ann['bbox']
                )
                examples.append(example)
                
        # Split and save
        return self._save_splits(examples, output_path)
    
    def process_rag_qa(
        self,
        json_path: str,
        output_path: str
    ) -> Tuple[int, int]:
        """
        Process RAG Q&A pairs into training data.
        
        JSON format:
        [
            {"question": "...", "answer": "...", "topic": "..."},
            ...
        ]
        """
        logger.info(f"Processing RAG Q&A from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
            
        examples = []
        for item in qa_data:
            example = self._create_rag_example(
                item['question'],
                item['answer'],
                item.get('topic', 'medical')
            )
            examples.append(example)
            
        return self._save_splits(examples, output_path)
    
    def _create_detection_example(
        self,
        image_path: str,
        class_name: str,
        bbox: List[float]
    ) -> Dict[str, Any]:
        """Create a detection training example"""
        
        # Get tool info
        tool_info = self.TOOL_MAPPING.get(
            class_name.lower(),
            {"action": "GroundingDINO", "prompt": class_name}
        )
        
        # Generate question
        question = random.choice(self.QUESTION_TEMPLATES)
        question = question.replace("{target}", class_name)
        
        # Generate plan JSON (4-field format)
        plan = {
            "thought": f"Cần kiểm tra {class_name} trong ảnh X-quang này.",
            "tool": "Vision",
            "action": tool_info["action"],
            "action_input": {
                "prompt": tool_info["prompt"]
            }
        }
        
        # Create conversation
        return {
            "image": image_path,
            "ground_truth_bbox": bbox,  # For RL training
            "conversations": [
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}
            ]
        }
    
    def _create_rag_example(
        self,
        question: str,
        answer: str,
        topic: str
    ) -> Dict[str, Any]:
        """Create a RAG training example (no image)"""
        
        # For theory questions, use RAG tool (4-field format)
        plan = {
            "thought": f"Đây là câu hỏi lý thuyết về {topic}. Cần sử dụng RAG.",
            "tool": "Knowledge",
            "action": "Medical_RAG",
            "action_input": {
                "query": question
            }
        }
        
        return {
            "image": None,  # No image for theory
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
                {"role": "user", "content": "Kết quả RAG cho thấy gì?"},
                {"role": "assistant", "content": answer}
            ]
        }
    
    def _save_splits(
        self,
        examples: List[Dict],
        output_path: str
    ) -> Tuple[int, int]:
        """Split and save data to train/val files"""
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Shuffle
        random.shuffle(examples)
        
        # Split
        split_idx = int(len(examples) * self.config.train_split)
        train_data = examples[:split_idx]
        val_data = examples[split_idx:]
        
        # Save train
        with open(output_path / "train.jsonl", 'w', encoding='utf-8') as f:
            for ex in train_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                
        # Save val
        with open(output_path / "val.jsonl", 'w', encoding='utf-8') as f:
            for ex in val_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                
        logger.info(f"Saved {len(train_data)} train, {len(val_data)} val to {output_path}")
        return len(train_data), len(val_data)
    
    def create_curriculum_splits(
        self,
        examples: List[Dict],
        output_path: str
    ) -> Dict[str, int]:
        """
        Create curriculum learning splits (for RL training).
        
        Levels:
        1. Easy: Single-turn, explicit targets
        2. Medium: Multi-turn with verification
        3. Hard: Complex reasoning required
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Classify by complexity
        easy, medium, hard = [], [], []
        
        for ex in examples:
            convs = ex.get('conversations', [])
            num_turns = len([c for c in convs if c['role'] == 'assistant'])
            
            if num_turns == 1:
                easy.append(ex)
            elif num_turns <= 3:
                medium.append(ex)
            else:
                hard.append(ex)
                
        # Save each level
        counts = {}
        for level, data in [('easy', easy), ('medium', medium), ('hard', hard)]:
            path = output_path / f"curriculum_{level}.jsonl"
            with open(path, 'w', encoding='utf-8') as f:
                for ex in data:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            counts[level] = len(data)
            
        logger.info(f"Curriculum splits: {counts}")
        return counts


# CLI for data processing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process medical data for training")
    parser.add_argument("--type", choices=["vindr", "rag", "both"], default="both")
    parser.add_argument("--csv", type=str, default="data/annotations.csv")
    parser.add_argument("--images", type=str, default="data/images")
    parser.add_argument("--rag", type=str, default="data/rag_qa.json")
    parser.add_argument("--output", type=str, default="data/sft_dataset")
    
    args = parser.parse_args()
    
    processor = MedicalDataProcessor()
    
    if args.type in ["vindr", "both"]:
        if Path(args.csv).exists():
            processor.process_vindr_cxr(args.csv, args.images, args.output)
            
    if args.type in ["rag", "both"]:
        if Path(args.rag).exists():
            processor.process_rag_qa(args.rag, args.output)
