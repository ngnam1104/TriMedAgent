from gradio.helpers import Examples
import argparse
import base64
from collections import defaultdict
import copy
import datetime
from functools import partial
import json
import os
import torch
from pathlib import Path
import cv2
import numpy as np
import re
import time
from io import BytesIO
from PIL import Image
from PIL import Image as _Image  # using _ to minimize namespace pollution

import gradio as gr
from gradio import processing_utils, utils
from gradio_client import utils as client_utils

import requests

from llava.conversation import (default_conversation, conv_templates,
                                SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
                         violates_moderation, moderation_msg)
import hashlib
from llava.serve.utils import annotate_xyxy, show_mask

import pycocotools.mask as mask_util

R = partial(round, ndigits=2)



class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch",
                         type='pil', interactive=True, **kwargs)
        # super().__init__(source="upload", tool="boxes", type='pil', interactive=True, **kwargs)

    def preprocess(self, x):
        # import ipdb; ipdb.set_trace()

        # a hack to get the mask
        if isinstance(x, str):
            im = processing_utils.decode_base64_to_image(x)
            w, h = im.size
            # a mask, array, uint8
            mask_np = np.zeros((h, w, 4), dtype=np.uint8)
            # to pil
            mask_pil = Image.fromarray(mask_np, mode='RGBA')
            # to base64
            mask_b64 = processing_utils.encode_pil_to_base64(mask_pil)
            x = {
                'image': x,
                'mask': mask_b64
            }

        res = super().preprocess(x)
        # arr -> PIL
        # res['image'] = Image.fromarray(res['image'])
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        return res


def get_mask_bbox(mask_img: Image):
    # convert to np array
    mask = np.array(mask_img)[..., 0]

    # check if has masks
    if mask.sum() == 0:
        return None

    # get coords
    coords = np.argwhere(mask > 0)

    # calculate bbox
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # get h and w
    h, w = mask.shape[:2]

    # norm to [0, 1]
    x0, y0, x1, y1 = R(x0 / w), R(y0 / h), R(x1 / w), R(y1 / h)
    return [x0, y0, x1, y1]


def plot_boxes(image: Image, res: dict) -> Image:
    boxes = torch.Tensor(res["boxes"])
    logits = torch.Tensor(res["logits"]) if 'logits' in res else None
    phrases = res["phrases"] if 'phrases' in res else None
    image_source = np.array(image)
    annotated_frame = annotate_xyxy(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    return Image.fromarray(annotated_frame)


def plot_masks(image: Image, res: dict) -> Image:
    masks_rle = res["masks_rle"]
    for mask_rle in masks_rle:
        mask = mask_util.decode(mask_rle)
        mask = torch.Tensor(mask)
        image = show_mask(mask, image)
    return image


def plot_points(image: Image, res: dict) -> Image:
    points = torch.Tensor(res["points"])
    point_labels = torch.Tensor(res["point_labels"])

    points = np.array(points)
    point_labels = np.array(point_labels)
    annotated_frame = np.array(image)
    h, w = annotated_frame.shape[:2]
    for i in range(points.shape[1]):
        color = (0, 255, 0) if point_labels[0, i] == 1 else (0, 0, 255)
        annotated_frame = cv2.circle(annotated_frame, (int(
            points[0, i, 0] * w), int(points[0, i, 1] * h)), 5, color, -1)
    return Image.fromarray(annotated_frame)


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "MMedAgent Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

R = partial(round, ndigits=2)

def b64_encode(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str

def get_worker_addr(controller_addr, worker_name):
    # get grounding dino addr
    if worker_name.startswith("http"):
        sub_server_addr = worker_name
    else:
        controller_addr = controller_addr
        ret = requests.post(controller_addr + "/refresh_all_workers")
        assert ret.status_code == 200
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        # print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": worker_name}
        )
        sub_server_addr = ret.json()["address"]
    # print(f"worker_name: {worker_name}")
    return sub_server_addr

try:
    with open("serve/labels.json", "r") as f:
        TRIMED_CONFIG = json.load(f)
except Exception as e:
    print(f"TriMedAgent Warning: Could not load serve/labels.json. Using defaults. Error: {e}")
    TRIMED_CONFIG = {
        "triage_labels": ["Chest X-ray", "Brain MRI", "Abdominal CT", "Histopathology", "Gross pathology"],
        "gatekeeper_prompts": {
            "positive": "Pathological finding",
            "negative": "Normal tissue"
        },
        "thresholds": {
            "triage_confidence": 0.5,
            "gatekeeper_confidence": 0.6,
            "low_confidence_fallback": 0.3
        },
        "conditional_execution": {
            "enabled": True,
            "skip_detection_modalities": ["Histopathology", "Dermoscopy"],
            "tool_mapping": {"default": ["grounding_dino"]}
        },
        "prompt_templates": {
            "triage_context": "[System Context: The input image is identified as {modality} with {confidence:.0%} confidence.]",
            "low_confidence_context": "[System Context: Image modality unclear.]"
        }
    }

# ================= TRIMEDAGENT HELPER FUNCTIONS =================

def get_threshold(key, default=0.6):
    """Lấy threshold từ config, có fallback."""
    return TRIMED_CONFIG.get("thresholds", {}).get(key, default)

def get_recommended_tools(modality):
    """Lấy danh sách tools phù hợp với modality."""
    tool_mapping = TRIMED_CONFIG.get("conditional_execution", {}).get("tool_mapping", {})
    return tool_mapping.get(modality, tool_mapping.get("default", ["grounding_dino"]))

def should_skip_detection(modality):
    """Kiểm tra xem có nên bỏ qua detection tool không."""
    if not TRIMED_CONFIG.get("conditional_execution", {}).get("enabled", False):
        return False
    skip_list = TRIMED_CONFIG.get("conditional_execution", {}).get("skip_detection_modalities", [])
    return modality in skip_list

def build_triage_context(modality, confidence):
    """Tạo context string để inject vào prompt."""
    templates = TRIMED_CONFIG.get("prompt_templates", {})
    threshold = get_threshold("triage_confidence", 0.5)
    
    if confidence >= threshold:
        template = templates.get("triage_context", 
            "[System Context: The input image is identified as {modality} with {confidence:.0%} confidence.]")
        tools = get_recommended_tools(modality)
        return template.format(modality=modality, confidence=confidence, tools=", ".join(tools))
    else:
        template = templates.get("low_confidence_context",
            "[System Context: Image modality unclear (best guess: {modality}, conf: {confidence:.0%}).]")
        return template.format(modality=modality, confidence=confidence)

def crop_image_base64(image_base64, box):
    """Cắt ảnh từ chuỗi base64 dựa trên bounding box [x1, y1, x2, y2] (0-1)."""
    try:
        img_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = box
        left, top, right, bottom = max(0, int(x1*w)), max(0, int(y1*h)), min(w, int(x2*w)), min(h, int(y2*h))
        cropped_img = img.crop((left, top, right, bottom))
        buffered = BytesIO()
        cropped_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"TriMedAgent Error - Cropping image: {e}")
        return None

def verify_boxes_with_gatekeeper(controller_url, image_base64, boxes, target_label, 
                                  logits=None, phrases=None, masks_rle=None):
    """
    Gửi các vùng ảnh (boxes) đi kiểm tra lại với BiomedCLIP.
    Trả về các boxes đã lọc VÀ các list liên quan (logits, phrases, masks) được đồng bộ.
    
    Returns:
        dict: {
            'boxes': filtered_boxes,
            'logits': filtered_logits (or None),
            'phrases': filtered_phrases (or None),
            'masks_rle': filtered_masks (or None),
            'valid_indices': list of kept indices,
            'rejected_indices': list of dropped indices
        }
    """
    result = {
        'boxes': [],
        'logits': [] if logits else None,
        'phrases': [] if phrases else None,
        'masks_rle': [] if masks_rle else None,
        'valid_indices': [],
        'rejected_indices': []
    }
    
    try:
        worker_addr = get_worker_addr(controller_url, "BiomedClip")
    except Exception as e:
        print(f"TriMedAgent Warning: BiomedClip worker not found. Skipping. Error: {e}")
        # Trả về nguyên gốc
        return {
            'boxes': boxes,
            'logits': logits,
            'phrases': phrases,
            'masks_rle': masks_rle,
            'valid_indices': list(range(len(boxes))),
            'rejected_indices': []
        }

    pos_label = f"{TRIMED_CONFIG['gatekeeper_prompts']['positive']} of {target_label}"
    neg_label = TRIMED_CONFIG['gatekeeper_prompts']['negative']
    gate_labels = [pos_label, neg_label]
    gate_threshold = get_threshold("gatekeeper_confidence", 0.6)

    print(f"TriMedAgent: Gatekeeper checking {len(boxes)} boxes (threshold={gate_threshold})")

    for i, box in enumerate(boxes):
        cropped_b64 = crop_image_base64(image_base64, box)
        if not cropped_b64:
            # Lỗi crop -> giữ lại an toàn
            result['valid_indices'].append(i)
            result['boxes'].append(box)
            if logits: result['logits'].append(logits[i] if i < len(logits) else 0)
            if phrases: result['phrases'].append(phrases[i] if i < len(phrases) else "")
            if masks_rle: result['masks_rle'].append(masks_rle[i] if i < len(masks_rle) else None)
            continue

        payload = {"image": cropped_b64, "labels": gate_labels}
        try:
            res = requests.post(worker_addr + "/worker_generate", json=payload, timeout=5).json()
            pred_label = res.get('prediction', '')
            confidence = res.get('confidence', 0.0)
            
            if pred_label == pos_label and confidence >= gate_threshold:
                result['valid_indices'].append(i)
                result['boxes'].append(box)
                if logits: result['logits'].append(logits[i] if i < len(logits) else 0)
                if phrases: result['phrases'].append(phrases[i] if i < len(phrases) else "")
                if masks_rle: result['masks_rle'].append(masks_rle[i] if i < len(masks_rle) else None)
                print(f"  [KEEP] Box {i} accepted. Conf: {confidence:.2f}")
            else:
                result['rejected_indices'].append(i)
                print(f"  [DROP] Box {i} rejected. Pred: {pred_label}, Conf: {confidence:.2f}")
        except Exception as e:
            # Fail-safe: giữ lại
            print(f"  [WARN] Box {i} - API error, keeping. Error: {e}")
            result['valid_indices'].append(i)
            result['boxes'].append(box)
            if logits: result['logits'].append(logits[i] if i < len(logits) else 0)
            if phrases: result['phrases'].append(phrases[i] if i < len(phrases) else "")
            if masks_rle: result['masks_rle'].append(masks_rle[i] if i < len(masks_rle) else None)

    print(f"TriMedAgent: Gatekeeper result - {len(result['boxes'])}/{len(boxes)} boxes kept")
    return result

# ================= TRIMEDAGENT TRIAGE FUNCTION =================

def run_visual_triage(controller_url, image_base64):
    """
    Chạy Visual Triage trên ảnh đầu vào.
    
    Returns:
        dict: {
            'modality': detected modality string,
            'confidence': float,
            'all_predictions': dict of all labels -> scores,
            'recommended_tools': list of tool names,
            'should_skip_detection': bool,
            'context_prompt': string to inject
        }
    """
    result = {
        'modality': 'Medical Image',
        'confidence': 0.0,
        'all_predictions': {},
        'recommended_tools': ['grounding_dino'],
        'should_skip_detection': False,
        'context_prompt': ''
    }
    
    try:
        triage_worker_addr = get_worker_addr(controller_url, "BiomedClip")
        
        triage_payload = {
            "image": image_base64,
            "labels": TRIMED_CONFIG.get("triage_labels", [])
        }
        
        triage_res = requests.post(
            triage_worker_addr + "/worker_generate", 
            json=triage_payload, 
            timeout=5
        ).json()
        
        result['modality'] = triage_res.get('prediction', 'Medical Image')
        result['confidence'] = triage_res.get('confidence', 0.0)
        result['all_predictions'] = triage_res.get('all_predictions', {})
        result['recommended_tools'] = get_recommended_tools(result['modality'])
        result['should_skip_detection'] = should_skip_detection(result['modality'])
        result['context_prompt'] = build_triage_context(result['modality'], result['confidence'])
        
        print(f"TriMedAgent Triage: {result['modality']} (conf: {result['confidence']:.2%})")
        print(f"  -> Recommended tools: {result['recommended_tools']}")
        print(f"  -> Skip detection: {result['should_skip_detection']}")
        
    except Exception as e:
        print(f"TriMedAgent Error - Triage failed: {e}")
        result['context_prompt'] = "[System Context: Triage unavailable. Proceeding with general analysis.]"
    
    return result

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(
        LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(
                value=model, visible=True)

    state = default_conversation.copy()
    return (state,
            dropdown_update,
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True),
            gr.Accordion.update(visible=True))


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    return (state, gr.Dropdown.update(
        choices=models,
        value=models[0] if len(models) > 0 else ""),
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
        gr.Accordion.update(visible=True))


def change_debug_state(state, with_debug_parameter_from_state, request: gr.Request):
    logger.info(f"change_debug_state. ip: {request.client.host}")
    print("with_debug_parameter_from_state: ", with_debug_parameter_from_state)
    with_debug_parameter_from_state = not with_debug_parameter_from_state

    # modify the text on debug_btn
    debug_btn_value = "Show Progress" if not with_debug_parameter_from_state else "Hide Progress"

    debug_btn_update = gr.Button.update(
        value=debug_btn_value,
    )
    state_update = with_debug_parameter_from_state
    return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), "", None) + (debug_btn_update, state_update)


def add_text(state, text, image_dict, ref_image_dict, image_process_mode, with_debug_parameter_from_state, request: gr.Request):
    # dict_keys(['image', 'mask'])
    if image_dict is not None:
        image = image_dict['image']
    else:
        image = None
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        state = default_conversation.copy()

        # a hack, for mask
        sketch_mask = image_dict['mask']
        if sketch_mask is not None:
            text = (text[0], text[1], text[2], sketch_mask)
            # check if visual prompt is used
            bounding_box = get_mask_bbox(sketch_mask)
            if bounding_box is not None:
                text_input_new = text[0] + f"\nInput box: {bounding_box}"
                text = (text_input_new, text[1], text[2], text[3])
                
        if ref_image_dict is not None:
            # text = (text[0], text[1], text[2], text[3], {
            #     'ref_image': ref_image_dict['image'],
            #     'ref_mask': ref_image_dict['mask']
            # })
            state.reference_image = b64_encode(ref_image_dict['image'])
            state.reference_mask = b64_encode(ref_image_dict['mask'])

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), "", None, None) + (disable_btn,) * 6


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, with_debug_parameter_from_state, api_key, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (no_change_btn,) * 6
        return

    if len(state.messages) == state.offset + 2:
        # # First round of conversation

        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower() and 'tools' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        print("template_name: ", template_name)

        # # hack:
        # # template_name = "multimodal_tools"
        # # import ipdb; ipdb.set_trace()
        # # image_name = [hashlib.md5(image.tobytes()).hexdigest() for image in state.get_images(return_pil=True)][0]

        new_state = conv_templates[template_name].copy()

        # if len(new_state.roles) == 2:
        #     new_state.roles = tuple(list(new_state.roles) + ["system"])
        # new_state.append_message(new_state.roles[2], f"receive an image with name `{image_name}.jpg`")

        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        
        # for reference image
        new_state.reference_image = getattr(state, 'reference_image', None)
        new_state.reference_mask = getattr(state, 'reference_mask', None)
        
        # update
        state = new_state
        
        print("Messages：", state.messages)

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
                        json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()
    # import ipdb; ipdb.set_trace()

# ================= TRIMEDAGENT STEP 1: INTELLIGENT TRIAGE =================
    # Chỉ chạy Triage nếu có ảnh trong lượt chat hiện tại
    triage_result = None  # Lưu kết quả Triage để dùng cho Conditional Execution
    current_images = state.get_images()
    if len(current_images) > 0:
        print("TriMedAgent: Executing Visual Triage...")
        triage_result = run_visual_triage(args.controller_url, current_images[0])
        
        # Lưu triage result vào state để dùng sau (cho Gatekeeper)
        state.triage_result = triage_result
        
        # Inject context vào prompt
        if triage_result['context_prompt']:
            prompt = triage_result['context_prompt'] + "\n" + prompt
            print(f"TriMedAgent: Context injected into prompt")
    
    # Save images
    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest()
                      for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(
            LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)
    # import ipdb; ipdb.set_trace()

    # Make requests
    pload = {
        "model": model_selector,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": state.get_images(),
        "openai_key": api_key  
    }
    logger.info(f"==== request ====\n{pload}\n==== request ====")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,) * 6

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=10)
        # import ipdb; ipdb.set_trace()
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,) * 6
                else:
                    output = data["text"] + \
                        f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        print("error: ", e)
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
        return

    # remove the cursor
    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (enable_btn,) * 6

    # check if we need tools
    model_output_text = state.messages[-1][1]
    # import ipdb; ipdb.set_trace()
    print("model_output_text: ", model_output_text,
          "Now we are going to parse the output.")
    # parse the output

    # import ipdb; ipdb.set_trace()

    try:
        pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
        matches = re.findall(pattern, model_output_text, re.DOTALL)
        # import ipdb; ipdb.set_trace()
        if len(matches) > 0:
            # tool_cfg = json.loads(matches[0][1].strip())
            try:
                tool_cfg = json.loads(matches[0][1].strip())
            except Exception as e:
                tool_cfg = json.loads(
                    matches[0][1].strip().replace("\'", "\""))
            print("tool_cfg:", tool_cfg)
        else:
            tool_cfg = None
    except Exception as e:
        logger.info(f"Failed to parse tool config: {e}")
        tool_cfg = None

    # run tool augmentation
    print("trigger tool augmentation with tool_cfg: ", tool_cfg)
    if tool_cfg is not None and len(tool_cfg) > 0:
        assert len(
            tool_cfg) == 1, "Only one tool is supported for now, but got: {}".format(tool_cfg)
        api_name = tool_cfg[0]['API_name']
        print(f"API NAME: {api_name}")
        tool_cfg[0]['API_params'].pop('image', None)
        images = state.get_raw_images()
        if len(images) > 0:
            image = images[0]
        else:
            image = None
        api_paras = {
            'image': image,
            "prompt": prompt,
            "box_threshold": 0.3,
            "text_threshold": 0.25,
            "openai_key": api_key,
            **tool_cfg[0]['API_params']
        }
        if api_name in ['inpainting']:
            api_paras['mask'] = getattr(state, 'mask_rle', None)
        if api_name in ['openseed', 'controlnet']:
            if api_name == 'controlnet':
                api_paras['mask'] = getattr(state, 'image_seg', None)
            api_paras['mode'] = api_name
            api_name = 'controlnet'
        if api_name == 'seem':
            reference_image = getattr(state, 'reference_image', None)
            reference_mask = getattr(state, 'reference_mask', None)
            api_paras['refimg'] = reference_image
            api_paras['refmask'] = reference_mask
            # extract ref image and mask
            

        # import ipdb; ipdb.set_trace()
        tool_worker_addr = get_worker_addr(controller_url, api_name)
        print("tool_worker_addr: ", tool_worker_addr)
        tool_response = requests.post(
            tool_worker_addr + "/worker_generate",
            headers=headers,
            json=api_paras,
        ).json()
        tool_response_clone = copy.deepcopy(tool_response)
        print("tool_response: ", tool_response)

        # clean up the response
        masks_rle = None
        edited_image = None
        image_seg = None  # for openseed
        iou_sort_masks = None
        if 'boxes' in tool_response:
            try:
                tool_response['boxes'] = [[R(_b) for _b in bb]
                                        for bb in tool_response['boxes']]
            except:
                pass
        if 'logits' in tool_response:
            try:
                tool_response['logits'] = [R(_l) for _l in tool_response['logits']]
            except:
                pass
        if 'scores' in tool_response:
            try:
                tool_response['scores'] = [R(_s) for _s in tool_response['scores']]
            except:
                pass
        if "masks_rle" in tool_response:
            masks_rle = tool_response.pop("masks_rle")
        if "edited_image" in tool_response:
            edited_image = tool_response.pop("edited_image")
        if "size" in tool_response:
            try:
                _ = tool_response.pop("size")
            except:
                pass
        if api_name == "easyocr":
            _ = tool_response.pop("boxes")
            _ = tool_response.pop("scores")
        if "retrieval_results" in tool_response:
            tool_response['retrieval_results'] = [
                {'caption': i['caption'], 'similarity': R(i['similarity'])}
                for i in tool_response['retrieval_results']
            ]
        if "image_seg" in tool_response:
            image_seg = tool_response.pop("image_seg")
        if "iou_sort_masks" in tool_response:
            iou_sort_masks = tool_response.pop("iou_sort_masks")
        if len(tool_response) == 0:
            tool_response['message'] = f"The {api_name} has processed the image."
        # hack
        if masks_rle is not None:
            state.mask_rle = masks_rle[0]
        if image_seg is not None:
            state.image_seg = image_seg

        # if edited_image is not None:
        #     edited_image

        # build new response
        new_response = f"{api_name} model outputs: {tool_response}\n\n"
        first_question = state.messages[-2][-1]
        if isinstance(first_question, tuple):
            first_question = first_question[0].replace("<image>", "")
        first_question = first_question.strip()

        # add new response to the state
        state.append_message(state.roles[0],
                             new_response +
                             "Please summarize the model outputs and answer my first question: {}".format(
                                 first_question)
                             )
        state.append_message(state.roles[1], None)

        # Construct prompt
        prompt2 = state.get_prompt()

        # Make new requests
        pload = {
            "model": model_name,
            "prompt": prompt2,
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_new_tokens), 1536),
            "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
            "images": f'List of {len(state.get_images())} images: {all_image_hash}',
             "openai_key": api_key
        }
        logger.info(f"==== request ====\n{pload}")
        pload['images'] = state.get_images()

        state.messages[-1][-1] = "▌"
        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,) * 6

        try:
            # Stream output
            response = requests.post(worker_addr + "/worker_generate_stream",
                                     headers=headers, json=pload, stream=True, timeout=10)
            # import ipdb; ipdb.set_trace()
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        output = data["text"][len(prompt2):].strip()
                        state.messages[-1][-1] = output + "▌"
                        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn,) * 6
                    else:
                        output = data["text"] + \
                            f" (error_code: {data['error_code']})"
                        state.messages[-1][-1] = output
                        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
                        return
                    time.sleep(0.03)
        except requests.exceptions.RequestException as e:
            state.messages[-1][-1] = server_error_msg
            yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
            return

        # remove the cursor
        state.messages[-1][-1] = state.messages[-1][-1][:-1]

        # add image(s)
        if edited_image is not None:
            edited_image_pil = Image.open(
                BytesIO(base64.b64decode(edited_image))).convert("RGB")
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil, "Crop")
        if image_seg is not None:
            edited_image_pil = Image.open(
                BytesIO(base64.b64decode(image_seg))).convert("RGB")
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil, "Crop")
        if iou_sort_masks is not None:
            assert isinstance(
                iou_sort_masks, list), "iou_sort_masks should be a list, but got: {}".format(iou_sort_masks)
            edited_image_pil_list = [Image.open(
                BytesIO(base64.b64decode(i))).convert("RGB") for i in iou_sort_masks]
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil_list, "Crop")
        if api_name in ['grounding_dino', 'ram+grounding_dino', 'blip2+grounding_dino', 'grounding dino']:
            edited_image_pil = Image.open(
                BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
            # ================= TRIMEDAGENT STEP 3: GATEKEEPER VERIFICATION =================
            # Kiểm tra nếu tool trả về bounding boxes
            if 'boxes' in tool_response and len(tool_response['boxes']) > 0:
                print("TriMedAgent: Activating Gatekeeper for Detection...")
                
                # Lấy từ khóa tìm kiếm (Target Entity) từ output của DINO
                target_entity = "abnormality"
                if 'phrases' in tool_response and len(tool_response['phrases']) > 0:
                    target_entity = tool_response['phrases'][0]

                # Lấy ảnh gốc (Base64) để crop
                raw_image_b64 = state.get_images()[0]

                # Gọi hàm lọc Gatekeeper (mới - đồng bộ tất cả)
                gate_result = verify_boxes_with_gatekeeper(
                    args.controller_url, 
                    raw_image_b64, 
                    tool_response['boxes'], 
                    target_entity,
                    logits=tool_response.get('logits'),
                    phrases=tool_response.get('phrases')
                )
                
                # Cập nhật lại tool_response với dữ liệu đã lọc và đồng bộ
                tool_response['boxes'] = gate_result['boxes']
                if gate_result['logits'] is not None:
                    tool_response['logits'] = gate_result['logits']
                if gate_result['phrases'] is not None:
                    tool_response['phrases'] = gate_result['phrases']

            # ================= END GATEKEEPER =================
            edited_image_pil = plot_boxes(edited_image_pil, tool_response)
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil, "Crop")
        if api_name in ['grounding_dino+sam', 'grounded_sam', 'grounding dino + MedSAM']:
            edited_image_pil = Image.open(
                BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
            
            # ================= TRIMEDAGENT STEP 4: G-SEG GATEKEEPER =================
            # Logic này cần lọc đồng bộ cả Box và Mask
            if 'boxes' in tool_response and len(tool_response['boxes']) > 0:
                print("TriMedAgent: Activating Gatekeeper for Grounded-Seg...")
                
                raw_image_b64 = state.get_images()[0]
                
                # Lấy tên bệnh từ output (nếu có) hoặc mặc định
                target_entity = "abnormality"
                if 'phrases' in tool_response and len(tool_response['phrases']) > 0:
                    target_entity = tool_response['phrases'][0]
                
                # Lấy masks từ clone (code gốc dùng tool_response_clone cho mask)
                all_masks = tool_response_clone.get('masks_rle', [])
                
                # Gọi hàm Gatekeeper mới (đồng bộ cả mask)
                gate_result = verify_boxes_with_gatekeeper(
                    args.controller_url, 
                    raw_image_b64, 
                    tool_response['boxes'], 
                    target_entity,
                    logits=tool_response.get('logits'),
                    phrases=tool_response.get('phrases'),
                    masks_rle=all_masks
                )
                
                # Cập nhật tool_response
                tool_response['boxes'] = gate_result['boxes']
                if gate_result['logits'] is not None:
                    tool_response['logits'] = gate_result['logits']
                if gate_result['phrases'] is not None:
                    tool_response['phrases'] = gate_result['phrases']
                
                # Cập nhật masks (quan trọng cho MedSAM plot)
                if gate_result['masks_rle'] is not None:
                    tool_response_clone['masks_rle'] = gate_result['masks_rle']
                    
            # ================= END G-SEG GATEKEEPER =================
            
            edited_image_pil = plot_boxes(edited_image_pil, tool_response)
            edited_image_pil = plot_masks(
                edited_image_pil, tool_response_clone)
            state.messages[-1][-1] = (state.messages[-1]
                                      [-1], edited_image_pil, "Crop")
        if api_name in ['sam']:
            if 'points' in tool_cfg[0]['API_params']:
                edited_image_pil = Image.open(
                    BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
                edited_image_pil = plot_masks(
                    edited_image_pil, tool_response_clone)
                tool_response_clone['points'] = tool_cfg[0]['API_params']['points']
                tool_response_clone['point_labels'] = tool_cfg[0]['API_params']['point_labels']
                edited_image_pil = plot_points(
                    edited_image_pil, tool_response_clone)

                state.messages[-1][-1] = (state.messages[-1]
                                          [-1], edited_image_pil, "Crop")
            else:
                assert 'boxes' in tool_cfg[0]['API_params'], "not find 'boxes' in {}".format(
                    tool_cfg[0]['API_params'].keys())
                edited_image_pil = Image.open(
                    BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
                edited_image_pil = plot_boxes(edited_image_pil, tool_response)
                tool_response_clone['boxes'] = tool_cfg[0]['API_params']['boxes']
                edited_image_pil = plot_masks(
                    edited_image_pil, tool_response_clone)
                state.messages[-1][-1] = (state.messages[-1]
                                          [-1], edited_image_pil, "Crop")

        yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (enable_btn,) * 6

    finish_tstamp = time.time()
    logger.info(f"{output}")

    # models = get_model_list()

    # FIXME: disabled temporarily for image generation.
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(force_str=True),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


title_markdown = ("""
# 👨‍⚕️👩‍⚕️ MMedAgent: Learning to Use Medical Tools with Multi-modal Agent
[[Paper]](https://arxiv.org/abs/2407.02483) [[Code]](https://github.com/Wangyixinxin/MMedAgent)
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")


def build_demo(embed_mode):
    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", visible=False, container=False)
    with gr.Blocks(title="MMedAgent", theme=gr.themes.Base()) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = ImageMask()

                cur_dir = os.path.dirname(os.path.abspath(__file__))

                with gr.Accordion("Reference Image", open=False, visible=False) as ref_image_row:
                    gr.Markdown(
                        "The reference image is for some specific tools, like SEEM.")
                    ref_image_box = ImageMask()
                
                with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
                    image_process_mode = gr.Radio(
                        ["Crop", "Resize", "Pad"],
                        value="Crop",
                        label="Preprocess for non-square image")
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(
                        minimum=0, maximum=2048, value=512, step=64, interactive=True, label="Max output tokens",)
                    # with_debug_parameter_check_box = gr.Checkbox(label="With debug parameter", checked=args.with_debug_parameter)

                api_key_input = gr.Textbox(label="OpenAI API Key", placeholder="Enter your OpenAI API key", type="password")
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="MMedAgent Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=False)
                with gr.Row(visible=False) as button_row:
                    debug_btn = gr.Button(
                        value="Show Progress", interactive=True)
                    # import ipdb; ipdb.set_trace()
                if args.with_debug_parameter:
                    debug_btn.value = "Show Progress"
                with_debug_parameter_state = gr.State(
                    value=args.with_debug_parameter,
                )

        with gr.Row():
            with gr.Column():
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/cell_00040.bmp",
                        "Can you find and count how many cells are there in this image?"],
                    [f"{cur_dir}/examples/0007d316f756b3fa0baea2ff514ce945.jpg",
                        "Does this x-ray image show a sign of Cardiomegaly? Find the area."],
                    [f"{cur_dir}/examples/022_083_t1.jpg",
                    "Find if there is a tumor in this image."]
                ], inputs=[imagebox, textbox], label="Grounding Examples: ")
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/WORD_0072_0290.jpg",
                        "Can you locate and segment the kidneys, spleen, and liver in this 2D abdominal CT image?"],
                    [f"{cur_dir}/examples/MCUCXR_0075_0.png",
                        "Identify the lungs and segment them in this x-ray image."],
                ], inputs=[imagebox, textbox], label="Grounding + Segmentation Examples: ")
                
                
            with gr.Column():
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/0a4fbc9ade84a7abd1680eb8ba031a9d.jpg",
                    "What is the imaging modality used for this medical image?"],
                    [f"{cur_dir}/examples/27660471_f14-ott-9-5531.jpg",
                    "What is the specific type of histopathology depicted in the image?"],
                    [f"{cur_dir}/examples/32535614_f2-amjcaserep-21-e923356.jpg",
                    "Can you tell me the modality of this image?"]
                ], inputs=[imagebox, textbox], label="Image Modality Classification Examples: ")
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/chest.jpg",
                    "Can you generate a report based on this image?"]
                ], inputs=[imagebox, textbox], label="Medical Report Generation Examples: ")
                false_report = """
                               This case highlights a critical oversight in a complex medical situation. While the initial focus on the patient's syncope was well-executed, subsequent findings of a 4.9 cm aortic aneurysm with celiac artery involvement were not sufficiently followed up. 
                               Comprehensive imaging of the entire aorta at the discovery point was crucial and could have potentially led to a more favorable outcome. It should be noted that the radiologist conducting the pulmonary artery CT scan should have autonomously extended the imaging to the full aorta without needing further orders.
                               """

                gr.Examples(examples=[
                    ["I'm suffering from anorexia nervosa. Can you explain it and tell me how to deal with it? Please consult authoritative source."],
                    ["Analyze this report and retrieve information from an authoritative database:\n"+false_report]],
                    inputs=[textbox], label="Retrieval Augmented Generation Examples:"
                )




        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners

        textbox.submit(add_text, [state, textbox, imagebox, ref_image_box, image_process_mode, with_debug_parameter_state], 
                    [state, chatbot, textbox, imagebox, ref_image_box, debug_btn]
                    ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens, with_debug_parameter_state, api_key_input],
                        [state, chatbot, debug_btn])

        submit_btn.click(add_text, [state, textbox, imagebox, ref_image_box, image_process_mode, with_debug_parameter_state], 
                        [state, chatbot, textbox, imagebox, ref_image_box, debug_btn]
                    ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens, with_debug_parameter_state, api_key_input],
                        [state, chatbot, debug_btn])
        
        debug_btn.click(change_debug_state, [state, with_debug_parameter_state], [
                        state, chatbot, textbox, imagebox] + [debug_btn, with_debug_parameter_state])

        if args.model_list_mode == "once":
            demo.load(load_demo, [url_params], [state, model_selector,
                                                chatbot, textbox, submit_btn, button_row, parameter_row, ref_image_row],
                      _js=get_window_url_params)
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None, [state, model_selector,
                                                           chatbot, textbox, submit_btn, button_row, parameter_row, ref_image_row])
        else:
            raise ValueError(
                f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str,
                        default="http://localhost:20001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument("--model-list-mode", type=str, default="once",
                        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--with_debug_parameter", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()
    models = [i for i in models if 'llava' in i]

    logger.info(args)
    demo = build_demo(args.embed)
    _app, local_url, share_url = demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10,
                                            api_open=True).launch(
        server_name=args.host, server_port=args.port, share=args.share, debug=args.debug)
    print("Local URL: ", local_url)
    print("Share URL: ", share_url)
