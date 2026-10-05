"""Caricamento dei modelli, preparazione dell'input e calcolo di Grad-CAM++."""
import os

import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import timm
import torch

from .nl_config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def _resolve_target_layer(arch, model):
  """Restituisce (layer, nome leggibile). Il nome va riportato nella tesi."""
  if arch == "resnet50":
    return model.layer4[-1].conv3, "layer4[-1].conv3"
  # timm mobilevit_s: model.stages e' un nn.Sequential di stage; ogni stage e'
  # a sua volta un nn.Sequential di blocchi (senza attributo .blocks).
  last_stage = model.stages[-1]
  if hasattr(last_stage, "blocks"):
    layer, name = last_stage.blocks[-1], "stages[-1].blocks[-1]"
  elif isinstance(last_stage, torch.nn.Sequential) and len(last_stage) > 0:
    layer, name = last_stage[-1], "stages[-1][-1]"
  else:
    layer, name = last_stage, "stages[-1]"
  return layer, f"{name} ({type(layer).__name__})"


def load_model(arch, checkpoint_path, device):
  timm_name = {"resnet50": "resnet50", "mobilevit_s": "mobilevit_s"}[arch]
  if not checkpoint_path or not os.path.exists(checkpoint_path):
    # Gli script originali proseguivano con pesi casuali: qui ci si ferma.
    raise FileNotFoundError(f"[{arch}] checkpoint non trovato: {checkpoint_path}")
  model = timm.create_model(timm_name, pretrained=False, num_classes=1)
  state_dict = torch.load(checkpoint_path, map_location=device)
  if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]
  model.load_state_dict(state_dict, strict=True)
  model = model.to(device).eval()
  layer, name = _resolve_target_layer(arch, model)
  return model, [layer], name


def load_rgb(path, size=IMG_SIZE):
  """Legge l'immagine esattamente come il val_tf dei training.

  Image.open(...).convert("RGB"), poi transforms.Resize((size, size)) se
  size non e' None (su immagini PIL equivale al resize bilineare di Pillow).
  size=None mantiene la risoluzione nativa (DermalAugmentor di HAM).
  """
  img = Image.open(path).convert("RGB")
  if size is not None and img.size != (size, size):
    img = img.resize((size, size), Image.BILINEAR)
  return np.asarray(img, dtype=np.uint8).copy()


def to_tensor(rgb_uint8, device):
  x = rgb_uint8.astype(np.float32) / 255.0
  x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
  return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)


class CamRunner:
  """Predizione P(melanoma) e mappa Grad-CAM++ (normalizzata in [0,1])."""

  def __init__(self, arch, checkpoint_path, device):
    self.arch = arch
    self.model, target_layers, self.target_layer_name = load_model(
        arch, checkpoint_path, device
    )
    self.cam = GradCAMPlusPlus(model=self.model, target_layers=target_layers)
    # Unico logit: la mappa spiega sempre la classe "melanoma"
    self.targets = [ClassifierOutputTarget(0)]

  def __call__(self, tensor):
    with torch.no_grad():
      p = torch.sigmoid(self.model(tensor)).item()
    cam = self.cam(input_tensor=tensor, targets=self.targets)[0]
    return float(p), np.asarray(cam, dtype=np.float32)
