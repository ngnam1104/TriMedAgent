"""
A model worker executes the model.
MODIFIED FOR TRIMEDAGENT: Supports dynamic labels and returns confidence scores.
"""
import sys
import os
import argparse
import asyncio
import json
import time
import threading
import uuid
from io import BytesIO
import base64
from typing import List, Tuple, Union

import torch
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import requests
from PIL import Image

# OpenCLIP imports
from open_clip import create_model_from_pretrained, get_tokenizer

# MMedAgent specific imports
from serve.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from serve.utils import build_logger, pretty_print_semaphore

GB = 1 << 30

now_file_name = os.__file__
logdir = "logs/workers/"
os.makedirs(logdir, exist_ok=True)
logfile = os.path.join(logdir, f"{now_file_name}.log")

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(now_file_name, logfile)
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        model_names,
        device,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.device = device
        
        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

        # Load BiomedCLIP Model
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        
        self.model.to(self.device)
        self.model.eval()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}. "
            f"worker_id: {worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def load_image(self, image_path: str):
        if os.path.exists(image_path):
            image_source = Image.open(image_path).convert("RGB")
        else:
            # base64 coding
            try:
                image_source = Image.open(BytesIO(base64.b64decode(image_path))).convert("RGB")
            except Exception as e:
                logger.error(f"Error loading image from base64: {e}")
                return None
        return image_source

    def predict(self, image, labels):
        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize labels
        template = 'the photo can be classified as '
        texts = self.tokenizer([template + l for l in labels], context_length=256).to(self.device)
        
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(image_tensor, texts)
            logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
            
            # Convert to list
            probs = logits.squeeze(0).cpu().numpy().tolist()
            
            # Find top prediction
            top_idx = torch.argmax(logits, dim=-1).item()
            top_label = labels[top_idx]
            top_prob = probs[top_idx]
            
            # Map all labels to probs
            all_probs = dict(zip(labels, probs))

        return top_label, top_prob, all_probs

    def generate_stream_func(self, params, device):
        # get inputs
        image_path = params["image"]
        
        # --- NEW LOGIC: Dynamic Labels ---
        # If 'labels' is provided in params, use it (for Triage/Gatekeeper)
        # Otherwise, use the default list (Backward compatibility)
        if "labels" in params and params["labels"]:
            labels = params["labels"]
        else:
            labels = [
                'adenocarcinoma histopathology',
                'brain MRI',
                'covid line chart',
                'diagnostic flowchart',
                'diagnostic scatter plot',
                'squamous cell carcinoma histopathology',
                'immunohistochemistry histopathology',
                'bone X-ray',
                'chest X-ray',
                'abdomen CT',
                'lung CT',
                'pie chart',
                'hematoxylin and eosin histopathology',
                'gross'
            ]

        # Load image
        image = self.load_image(image_path)
        if image is None:
            return {"error": "Image load failed"}

        # Run Prediction
        top_label, top_prob, all_probs = self.predict(image, labels)

        # --- NEW OUTPUT FORMAT ---
        # Return a dictionary with confidence scores
        return {
            "prediction": top_label,
            "confidence": top_prob,
            "all_predictions": all_probs
        }

    def generate_gate(self, params):
        try:
            # Call the main logic
            result = self.generate_stream_func(params, self.device)
            
            # Wrap in the expected response format if needed, or return directly
            # Controller/Web Server expects a JSON response
            return result

        except torch.cuda.OutOfMemoryError as e:
            return {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError, Exception) as e:
            logger.error(f"Error in generate_gate: {e}")
            return {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }


app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    return model_semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return background_tasks


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    output = worker.generate_gate(params)
    release_model_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/model_details")
async def model_details(request: Request):
    # BiomedCLIP context length is typically 256
    return {"context_length": 256}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21006)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21006")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:20001"
    )
    parser.add_argument(
        "--model-names",
        default="BiomedClip",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_names,
        args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")