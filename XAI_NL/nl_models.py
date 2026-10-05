"""Caricamento dei modelli, preparazione dell'input e calcolo di Grad-CAM++."""
import os

import cv2
import numpy as np
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
  """Legge un'immagine come RGB uint8 alla risoluzione dei modelli.

  Le immagini preelaborate sono gia' state ridimensionate in fase di
  preprocessing; se non lo sono si usa l'interpolazione bilineare di OpenCV,
  la stessa del training (Sezione 2.2.2). Il filtro gaussiano NON viene
  riapplicato: si assume gia' presente nelle immagini di data/processed.
  """
  bgr = cv2.imread(path, cv2.IMREAD_COLOR)
  if bgr is None:
    raise IOError(f"Impossibile leggere {path}")
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  if rgb.shape[:2] != (size, size):
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
  return rgb


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
