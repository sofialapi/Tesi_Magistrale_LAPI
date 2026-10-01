import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import timm
import torch
from torchvision import transforms


def get_model(arch_type, checkpoint_path=None, device="cuda"):
  if arch_type == "cnn":
    model = timm.create_model("resnet50", pretrained=False, num_classes=1)
    target_layers = [model.layer4[-1].conv3]
  elif arch_type == "hybrid":
    model = timm.create_model("mobilevit_s", pretrained=False, num_classes=1)
    if hasattr(model, "stages") and len(model.stages) > 0:
      last_stage = model.stages[-1]
      target_layers = (
          [last_stage.blocks[-1]]
          if hasattr(last_stage, "blocks")
          else [last_stage]
      )
    else:
      target_layers = [list(model.children())[-2]]

  if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"[{arch_type}] Caricamento pesi: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
      state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)

  model = model.to(device)
  model.eval()
  return model, target_layers


def preprocess_image(img_path, img_size=(224, 224)):
  raw_image = Image.open(img_path).convert("RGB").resize(img_size)
  rgb_img = np.float32(raw_image) / 255.0
  tf = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])
  tensor_img = tf(raw_image).unsqueeze(0)
  return rgb_img, tensor_img


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--csv_path",
      type=str,
      default="data/ham10000/ham10000_prepared.csv",
  )
  parser.add_argument(
      "--cnn_ckpt",
      type=str,
      default="outputs/checkpoints/best_cnn_ham10000.pth",
  )
  parser.add_argument(
      "--hybrid_ckpt",
      type=str,
      default="outputs/checkpoints/best_hybrid_ham10000.pth",
  )
  parser.add_argument("--output_dir", type=str, default="figure/ham10000/xai")
  parser.add_argument(
      "--threshold",
      type=float,
      default=0.50,
      help="Soglia decisionale (default 0.50 poiche pesata)",
  )
  parser.add_argument("--num_samples", type=int, default=6)
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  os.makedirs(args.output_dir, exist_ok=True)

  cnn_model, cnn_targets = get_model("cnn", args.cnn_ckpt, device)
  hybrid_model, hybrid_targets = get_model("hybrid", args.hybrid_ckpt, device)

  cam_cnn = GradCAMPlusPlus(model=cnn_model, target_layers=cnn_targets)
  cam_hybrid = GradCAMPlusPlus(model=hybrid_model, target_layers=hybrid_targets)
  cam_targets = [ClassifierOutputTarget(0)]

  df = pd.read_csv(args.csv_path)
  val_df = df[df["fold"] == 1].reset_index(drop=True)

  malignant = val_df[val_df["target"] == 1].head(args.num_samples // 2)
  benign = val_df[val_df["target"] == 0].head(
      args.num_samples - len(malignant)
  )
  samples = pd.concat([malignant, benign], ignore_index=True)

  for idx, row in samples.iterrows():
    path = row["filepath"]
    label = int(row["target"])
    img_id = row["image_id"]
    label_str = "Melanoma" if label == 1 else "Benigno"

    rgb_img, input_tensor = preprocess_image(path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
      p_cnn = torch.sigmoid(cnn_model(input_tensor)).item()
      p_hyb = torch.sigmoid(hybrid_model(input_tensor)).item()

    vis_cnn = show_cam_on_image(
        rgb_img,
        cam_cnn(input_tensor=input_tensor, targets=cam_targets)[0, :],
        use_rgb=True,
    )
    vis_hyb = show_cam_on_image(
        rgb_img,
        cam_hybrid(input_tensor=input_tensor, targets=cam_targets)[0, :],
        use_rgb=True,
    )

    esito_cnn = "Melanoma" if p_cnn >= args.threshold else "Benigno"
    esito_hyb = "Melanoma" if p_hyb >= args.threshold else "Benigno"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title(
        f"Immagine: {img_id}\nGround Truth: {label_str}",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].axis("off")

    color_cnn = "green" if esito_cnn == label_str else "red"
    axes[1].imshow(vis_cnn)
    axes[1].set_title(
        f"ResNet-50 (CNN)\nP(Melanoma): {p_cnn:.4f}\nEsito (tau={args.threshold}):"
        f" {esito_cnn}",
        fontsize=11,
        color=color_cnn,
    )
    axes[1].axis("off")

    color_hyb = "green" if esito_hyb == label_str else "red"
    axes[2].imshow(vis_hyb)
    axes[2].set_title(
        f"MobileViT-S (Hybrid)\nP(Melanoma): {p_hyb:.4f}\nEsito"
        f" (tau={args.threshold}): {esito_hyb}",
        fontsize=11,
        color=color_hyb,
    )
    axes[2].axis("off")

    plt.tight_layout()
    out_file = os.path.join(
        args.output_dir, f"ham_gradcam_{idx+1}_{label_str}.png"
    )
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[{idx+1}/{len(samples)}] Salvato: {out_file}")


if __name__ == "__main__":
  main()
