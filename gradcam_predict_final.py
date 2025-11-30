#!/usr/bin/env python3
"""
gradcam_predict_final.py

Procesa una imagen o un video:
- Detecta caras con tu FaceDetector (RetinaFace desde batch_face)
- Omite imagen/frame si no se detectan caras (registra en predicciones.txt)
- Para cada cara: recorta con padding, redimensiona, infiere, genera Grad-CAM++ y guarda overlay
- Crea carpeta Prediccion_<inputname>/overlay/ con 1.jpg,2.jpg,...
- Crea predicciones.txt con resultados, items omitidos y conteo final

Dependencias (en tu entorno): torch, torchvision, timm, opencv-python, pillow, numpy, batch_face (RetinaFace)
"""

import argparse
import json
import logging as log
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as T
from collections import Counter

# Tu import real del detector
from batch_face import RetinaFace

# --------------------
# FaceDetector (adaptado de tu código)
# --------------------
class FaceDetector:
    def __init__(self, device, batch_size, threshold=0.95, increment=0.1):
        """
        device: 'cpu' or 'cuda:0' style string
        batch_size: int for detector batching
        threshold: float or tuple(range) like (high, low)
        """
        self.device = device
        self.batch_size = batch_size
        self.threshold = threshold
        self.increment = increment
        # RetinaFace expects -1 for CPU or integer device index for GPU
        try:
            if isinstance(device, str) and device.lower().startswith("cpu"):
                dev_idx = -1
            elif isinstance(device, str) and device.lower().startswith("cuda"):
                parts = device.split(":")
                dev_idx = int(parts[1]) if len(parts) > 1 else 0
            else:
                dev_idx = -1
        except Exception:
            dev_idx = -1
        # instantiate retinaface from batch_face module
        try:
            self.detector = RetinaFace(dev_idx)
        except Exception as e:
            raise RuntimeError("No se pudo instanciar RetinaFace. Asegúrate de que 'batch_face.RetinaFace' está disponible. Error: " + str(e))

    @torch.no_grad()
    def _batch_detect(self, frames: List[np.ndarray], threshold: float):
        """
        frames: list of RGB uint8 numpy arrays
        returns: boxes, landmarks, scores  (lists aligned with frames)
        Each element in boxes is either [] or np.ndarray Nx4 of [l,t,r,b]
        """
        boxes, landmarks, scores = [], [], []
        for idx in range(0, len(frames), self.batch_size):
            batch_frames = frames[idx : idx + self.batch_size]
            batch_results = self.detector(batch_frames)  # expected list of lists per frame
            for frame_results in batch_results:
                # frame_results expected: list of (bbox, landmarks, score)
                valid_res = [res for res in frame_results if res[2] > threshold]
                if len(valid_res) > 0:
                    frame_boxes, frame_landmarks, frame_scores = list(zip(*valid_res))
                    boxes.append(np.vstack(frame_boxes) if len(frame_boxes) > 0 else [])
                    landmarks.append(np.vstack(frame_landmarks) if len(frame_landmarks) > 0 else [])
                    scores.append(list(frame_scores))
                else:
                    boxes.append([])
                    landmarks.append([])
                    scores.append([])
        return boxes, landmarks, scores

    @staticmethod
    def _is_continuous(scores):
        for sc in scores:
            if len(sc) == 0:
                return False
        return True

    @torch.no_grad()
    def __call__(self, frames: List[np.ndarray]):
        # frames: list of RGB images
        if isinstance(self.threshold, float):
            return self._batch_detect(frames, self.threshold)
        elif isinstance(self.threshold, (list, tuple)):
            assert self.threshold[0] > self.threshold[1] and len(self.threshold) == 2, "invalid threshold range"
            for thr in np.arange(self.threshold[0], self.threshold[1], step=-self.increment):
                boxes, lmks, scores = self._batch_detect(frames, thr)
                torch.cuda.empty_cache()
                if self._is_continuous(scores):
                    break
            return boxes, lmks, scores
        else:
            return self._batch_detect(frames, float(self.threshold))

# --------------------
# Grad-CAM++ implementacion robusta
# --------------------
class GradCAMpp:
    def __init__(self, model, target_layers, device):
        self.model = model.eval()
        self.device = device
        self.model.to(self.device)

        if isinstance(target_layers, (list, tuple)):
            self.target_layers = list(target_layers)
        else:
            self.target_layers = [target_layers]

        self.activations = {}
        self.gradients = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def make_forward(layer):
            def fh(module, inp, out):
                self.activations[id(module)] = out.detach()
            return fh

        def make_backward(layer):
            def bh(module, grad_input, grad_output):
                g = grad_output[0]
                if g is None:
                    return
                self.gradients[id(module)] = g.detach()
            return bh

        for layer in self.target_layers:
            self.handles.append(layer.register_forward_hook(make_forward(layer)))
            if hasattr(layer, "register_full_backward_hook"):
                def wrap_full(module):
                    def full_hook(mod, grad_input, grad_output):
                        if grad_output and grad_output[0] is not None:
                            self.gradients[id(mod)] = grad_output[0].detach()
                    return full_hook
                self.handles.append(layer.register_full_backward_hook(wrap_full(layer)))
            else:
                self.handles.append(layer.register_backward_hook(make_backward(layer)))

    def remove_hooks(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _clear(self):
        self.activations.clear()
        self.gradients.clear()

    def __call__(self, input_tensor, class_idx=None, retain_graph=False):
        """
        input_tensor: [B,C,H,W] on device
        returns: np.array [B,H,W] normalized 0..1
        """
        self._clear()
        input_tensor = input_tensor.to(self.device)

        outputs = self.model(input_tensor)  # [B, C]
        if class_idx is None:
            pred = outputs.argmax(dim=1)
        else:
            if isinstance(class_idx, int):
                pred = torch.tensor([class_idx] * outputs.size(0), device=self.device)
            else:
                pred = torch.tensor(class_idx, device=self.device)

        scores = outputs.gather(1, pred.view(-1,1)).squeeze()  # [B]
        self.model.zero_grad()
        scores.sum().backward(retain_graph=retain_graph)

        cams_per_layer = []
        for layer in self.target_layers:
            aid = id(layer)
            if aid not in self.activations or aid not in self.gradients:
                raise RuntimeError(f"Missing activation or gradient for layer {layer}")
            act = self.activations[aid]   # [B,C,H,W]
            grad = self.gradients[aid]    # [B,C,H,W]

            if act.dim() != grad.dim():
                raise RuntimeError(f"act/grad dim mismatch: {act.shape} vs {grad.shape}")
            if act.shape != grad.shape:
                raise RuntimeError(f"act/grad shape mismatch: {act.shape} vs {grad.shape}")

            b, k, u, v = act.size()
            grad2 = grad ** 2
            grad3 = grad ** 3
            eps = 1e-8
            global_sum = torch.sum(act * grad3, dim=(2,3), keepdim=True)
            denominator = 2.0 * grad2 + global_sum + eps
            alpha = grad2 / denominator
            relu_grad = F.relu(grad)
            weights = torch.sum(alpha * relu_grad, dim=(2,3))
            cam = torch.sum(weights.view(b,k,1,1) * act, dim=1)
            cam = F.relu(cam)
            cams_per_layer.append(cam.detach().cpu().numpy())

        cams_stack = np.stack(cams_per_layer, axis=0)
        cam_avg = np.mean(cams_stack, axis=0)

        out = []
        for i in range(cam_avg.shape[0]):
            cam_i = cam_avg[i]
            cam_i -= cam_i.min()
            if cam_i.max() > 0:
                cam_i = cam_i / cam_i.max()
            out.append(cam_i)
        return np.stack(out, axis=0)

# --------------------
# Util helpers
# --------------------
def scale_bbox(bbox, height, width, scale_factor):
    left, top, right, bottom = bbox
    size_bb = int(max(right - left, bottom - top) * scale_factor)
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    left = max(int(center_x - size_bb // 2), 0)
    top = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - left, size_bb)
    size_bb = min(height - top, size_bb)
    return left, top, left + size_bb, top + size_bb

def apply_bbox(image, bbox, scale_factor=None):
    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)
    bbox[bbox < 0] = 0
    bbox = np.around(bbox).astype(int)
    if scale_factor:
        bbox = scale_bbox(bbox, image.shape[0], image.shape[1], scale_factor)
    left, top, right, bottom = bbox
    face = image[top:bottom, left:right, :]
    return face

def overlay_cam_on_image(img_rgb, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    overlay = (heatmap.astype(np.float32) * alpha + img_rgb.astype(np.float32) * (1-alpha)).astype(np.uint8)
    return overlay

def safe_load_state_dict(model, ckpt):
    """Try several common checkpoint dict formats and fix module. prefix if needed"""
    if isinstance(ckpt, dict):
        if 'model_state' in ckpt:
            state = ckpt['model_state']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    # if keys start with module. strip
    new_state = {}
    for k, v in state.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_state[new_k] = v
    return new_state

def load_model_from_checkpoint(model_name, model_path, num_classes=5, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"[load_model] building {model_name} on {device}", flush=True)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(model_path, map_location=device)
    state = safe_load_state_dict(model, ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print("[load_model] done", flush=True)
    return model

# --------------------
# Processing routines (image / video) using FaceDetector
# --------------------
# --------------------
# process_image_file_with_detector (limita a max_faces)
# --------------------
def process_image_file_with_detector(img_path: Path, face_detector: FaceDetector, model, gradcam, device, transform, out_overlay_dir, results_list, img_size=224, pad_ratio=0.25, max_faces=1):
    pil = Image.open(img_path).convert("RGB")
    img_np = np.array(pil)
    # face_detector expects list of frames
    boxes_list, lmks_list, scores_list = face_detector([img_np])
    omitted = []
    boxes = boxes_list[0] if len(boxes_list) > 0 else []
    scores = scores_list[0] if len(scores_list) > 0 else []
    if boxes is None or len(boxes) == 0:
        print(f"No se detectó ninguna cara en la imagen {img_path.name}; se omite.", flush=True)
        omitted.append(img_path.name)
        return results_list, omitted

    # select top-k by score if requested
    if max_faces > 0 and len(scores) > 0:
        idxs = np.argsort(scores)[-max_faces:][::-1]  # indices of top scores
        sel_boxes = [boxes[i] for i in idxs]
    elif max_faces > 0 and len(scores) == 0:
        # fallback: take largest bounding boxes by area
        areas = [ (b[2]-b[0])*(b[3]-b[1]) for b in boxes ]
        idxs = np.argsort(areas)[-max_faces:][::-1]
        sel_boxes = [boxes[i] for i in idxs]
    else:
        sel_boxes = boxes  # max_faces == 0 -> process all

    crops = []
    for bbox in sel_boxes:
        face = apply_bbox(img_np, bbox, scale_factor=1+pad_ratio)
        face = cv2.resize(face, (img_size, img_size))
        crops.append(face)

    tensors = []
    for c in crops:
        tensors.append(transform(Image.fromarray(c)))
    x = torch.stack(tensors).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    for i, (c, p) in enumerate(zip(crops, probs)):
        pred = int(np.argmax(p))
        pred_prob = float(p[pred])
        xt = transform(Image.fromarray(c)).unsqueeze(0).to(device)
        cams = gradcam(xt, class_idx=pred)
        cam = cams[0]
        overlay = overlay_cam_on_image(c, cam, alpha=0.5)
        fname = f"{len(results_list)+1}.jpg"
        out_path = out_overlay_dir / fname
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        results_list.append((fname, pred, pred_prob, p.tolist()))
    return results_list, omitted


# --------------------
# process_video_file_with_detector (limita a max_faces)
# --------------------
def process_video_file_with_detector(video_path: Path, face_detector: FaceDetector, model, gradcam, device, transform, out_overlay_dir, fps=1, img_size=224, pad_ratio=0.25, max_faces=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = []
    frame_infos = []
    sec = 0.0
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000.0)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_infos.append((idx, int(sec)))
        sec += 1.0/fps
        idx += 1
    cap.release()

    if len(frames) == 0:
        return [], [f"video_no_frames"]

    # detect faces in batch
    boxes_list, lmks_list, scores_list = face_detector(frames)
    results = []
    omitted_frames = []
    saved_idx = 0

    for i, (frame_rgb, boxes, scores, info) in enumerate(zip(frames, boxes_list, scores_list, frame_infos)):
        frame_idx, sec_t = info
        if boxes is None or len(boxes) == 0:
            omitted_frames.append(f"frame_{frame_idx:06d}_t{sec_t}s")
            continue

        # choose top-k boxes by score or by area fallback
        if max_faces > 0 and len(scores) > 0:
            idxs = np.argsort(scores)[-max_faces:][::-1]
            sel_boxes = [boxes[j] for j in idxs]
        elif max_faces > 0 and len(scores) == 0:
            areas = [ (b[2]-b[0])*(b[3]-b[1]) for b in boxes ]
            idxs = np.argsort(areas)[-max_faces:][::-1]
            sel_boxes = [boxes[j] for j in idxs]
        else:
            sel_boxes = boxes

        crops = []
        for bbox in sel_boxes:
            face = apply_bbox(frame_rgb, bbox, scale_factor=1+pad_ratio)
            face = cv2.resize(face, (img_size, img_size))
            crops.append(face)

        tensors = []
        for c in crops:
            tensors.append(transform(Image.fromarray(c)))
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        for p_idx, (c, p) in enumerate(zip(crops, probs)):
            pred = int(np.argmax(p))
            pred_prob = float(p[pred])
            xt = transform(Image.fromarray(c)).unsqueeze(0).to(device)
            cams = gradcam(xt, class_idx=pred)
            cam = cams[0]
            overlay = overlay_cam_on_image(c, cam, alpha=0.5)
            saved_idx += 1
            fname = f"{saved_idx}.jpg"
            out_path = out_overlay_dir / fname
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            results.append((fname, pred, pred_prob, p.tolist(), f"frame_{frame_idx:06d}_t{sec_t}s"))
    return results, omitted_frames


# --------------------
# Main CLI
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to image or video")
    parser.add_argument("--model", required=True, help="path to .pth checkpoint")
    parser.add_argument("--model-name", required=True, help="timm model name used in training")
    parser.add_argument("--outroot", default=".", help="root folder to create Prediccion_*")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=1, help="frames per second for video sampling")
    parser.add_argument("--pad", type=float, default=0.25, help="face bbox padding ratio")
    parser.add_argument("--device", default=None, help="cuda or cpu; default auto")
    # detector-specific
    parser.add_argument("--detector-device", default=None, help="device string for face detector e.g. cpu or cuda:0")
    parser.add_argument("--detector-batch-size", type=int, default=8, help="batch size for face detector")
    parser.add_argument("--detector-threshold", default=0.95, help="threshold float or range 'high,low' e.g. 0.95 or 0.95,0.5")
    parser.add_argument("--detector-increment", type=float, default=0.1, help="increment step when using threshold range")
    parser.add_argument("--max-faces", type=int, default=1, help="max faces to process per frame; 0 = process all detections")

    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print("Input not found:", inp); sys.exit(1)

    model_device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Using device (model):", model_device, flush=True)

    # detector device string
    def device_to_detector_str(device):
        if isinstance(device, str):
            return device
        if device.type == "cpu":
            return "cpu"
        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            return f"cuda:{idx}"
        return "cpu"

    det_dev_str = args.detector_device if args.detector_device else device_to_detector_str(model_device)
    print("Face detector device:", det_dev_str, flush=True)

    # parse detector threshold (allow range)
    det_threshold = None
    if isinstance(args.detector_threshold, str) and "," in str(args.detector_threshold):
        parts = [float(x.strip()) for x in str(args.detector_threshold).split(",")]
        if len(parts) == 2:
            det_threshold = (parts[0], parts[1])
    else:
        det_threshold = float(args.detector_threshold)

    face_detector = FaceDetector(device=det_dev_str, batch_size=args.detector_batch_size, threshold=det_threshold, increment=args.detector_increment)

    # load model
    model = load_model_from_checkpoint(args.model_name, args.model, device=model_device)

    # find last conv layer(s)
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if len(conv_layers) == 0:
        raise RuntimeError("No Conv2d layers found in model")
    target_layers = conv_layers[-1:]
    print("Using target layers:", target_layers, flush=True)
    gradcam = GradCAMpp(model, target_layers, device=model_device)

    # transforms (val)
    transform = T.Compose([
        T.Resize(int(args.img_size * 1.15)),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # prepare output dirs (unique)
    safe_name = inp.stem.replace(" ", "_")
    outdir = Path(args.outroot) / f"Prediccion_{safe_name}"
    overlays_dir = outdir / "overlay"
    if outdir.exists():
        n = 1
        base = outdir
        while outdir.exists():
            outdir = Path(str(base) + f"_{n}")
            overlays_dir = outdir / "overlay"
            n += 1
    overlays_dir.mkdir(parents=True, exist_ok=True)

    results = []
    omitted_items = []
    if inp.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        print("Processing image:", inp, flush=True)
        results, omitted = process_image_file_with_detector(inp, face_detector, model, gradcam, model_device, transform, overlays_dir, results, img_size=args.img_size, pad_ratio=args.pad)
        omitted_items.extend(omitted)
    else:
        print("Processing video (sampling fps = {}):".format(args.fps), inp, flush=True)
        results, omitted_frames = process_video_file_with_detector(inp, face_detector, model, gradcam, model_device, transform, overlays_dir, fps=args.fps, img_size=args.img_size, pad_ratio=args.pad)
        omitted_items.extend(omitted_frames)

    # write TXT summary
    txt_path = outdir / "predicciones.txt"
    total = len(results)
    counts = Counter()

    if total == 0:
        with open(txt_path, "w") as f:
            f.write(f"Input: {inp.name}\n")
            f.write("No se detecta ninguna cara en el video/imagen.\n")
            if len(omitted_items) > 0:
                f.write("\nItems omitidos (sin caras):\n")
                for it in omitted_items:
                    f.write(f"{it}\n")
        try:
            if overlays_dir.exists() and len(list(overlays_dir.iterdir())) == 0:
                overlays_dir.rmdir()
        except Exception:
            pass
        print("No se detectó ninguna cara en la entrada. Mensaje guardado en", txt_path, flush=True)
        gradcam.remove_hooks()
        return

    with open(txt_path, "w") as f:
        f.write(f"Input: {inp.name}\n")
        f.write(f"Total crops processed: {total}\n\n")
        f.write("Per-image predictions (filename --- pred_class --- pred_prob --- [p0..pN] --- source_frame_if_video):\n")
        for item in results:
            if len(item) == 5:
                fname, pred, pred_prob, probs, source = item
            else:
                fname, pred, pred_prob, probs = item
                source = "image"
            f.write(f"{fname} --- {pred} --- {pred_prob:.6f} --- {np.array2string(np.array(probs), precision=4, separator=',')} --- {source}\n")
            counts[pred] += 1

        if len(omitted_items) > 0:
            f.write("\nItems omitidos (sin caras):\n")
            for it in omitted_items:
                f.write(f"{it}\n")

        f.write("\nCounts and percentages:\n")
        for cls, cnt in sorted(counts.items()):
            pct = cnt/total*100 if total>0 else 0.0
            f.write(f"Class {cls}: {cnt} images ({pct:.2f}%)\n")
        if total>0:
            majority = counts.most_common(1)[0]
            f.write(f"\nMajority vote: class {majority[0]} with {majority[1]} / {total} images ({majority[1]/total*100:.2f}%)\n")

    print("Saved overlays in:", overlays_dir, flush=True)
    print("Saved summary TXT:", txt_path, flush=True)

    gradcam.remove_hooks()

if __name__ == "__main__":
    main()
