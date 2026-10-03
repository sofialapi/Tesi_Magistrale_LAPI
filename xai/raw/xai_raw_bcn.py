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
            target_layers = [last_stage.blocks[-1]] if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0 else [last_stage]
        elif hasattr(model, "conv_1x1_exp"):
            target_layers = [model.conv_1x1_exp]
        else:
            target_layers = [list(model.children())[-2]]
    else:
        raise ValueError(f"Architettura non valida: {arch_type}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[{arch_type}] Caricamento pesi raw da: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} non trovato!")

    model = model.to(device)
    model.eval()
    return model, target_layers

def preprocess_image(img_path, img_size=(224, 224)):
    raw_image = Image.open(img_path).convert("RGB").resize(img_size)
    rgb_img = np.float32(raw_image) / 255.0
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_img = tf(raw_image).unsqueeze(0)
    return rgb_img, tensor_img

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM++ BCN20000 Raw")
    parser.add_argument("--data_dir", type=str, default="data/bcn20000")
    parser.add_argument("--cnn_ckpt", type=str, default="outputs/checkpoints/raw/best_BCN_raw_cnn.pth")
    parser.add_argument("--hybrid_ckpt", type=str, default="outputs/checkpoints/raw/best_BCN_raw_hybrid.pth")
    parser.add_argument("--output_dir", type=str, default="figure/raw/bcn20000/xai")
    parser.add_argument("--threshold", type=float, default=0.50, help="Soglia decisionale calibrata")
    parser.add_argument("--num_samples", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    cnn_model, cnn_targets = get_model("cnn", args.cnn_ckpt, device)
    hybrid_model, hybrid_targets = get_model("hybrid", args.hybrid_ckpt, device)

    cam_cnn = GradCAMPlusPlus(model=cnn_model, target_layers=cnn_targets)
    cam_hybrid = GradCAMPlusPlus(model=hybrid_model, target_layers=hybrid_targets)
    cam_targets = [ClassifierOutputTarget(0)]

    csv_path = os.path.join(args.data_dir, "train.csv")
    img_dir = os.path.join(args.data_dir, "train")
    df = pd.read_csv(csv_path)

    existing_files = {f: os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")}
    valid_rows = []
    for _, row in df.iterrows():
        base = str(row["image_name"]).replace(".jpg", "")
        candidates = [f"{base}.jpg", f"{base}_downsampled.jpg"]
        for c in candidates:
            if c in existing_files:
                valid_rows.append((existing_files[c], int(row["target"]), str(row["image_name"])))
                break

    malignant = [x for x in valid_rows if x[1] == 1][: args.num_samples // 2]
    benign = [x for x in valid_rows if x[1] == 0][: args.num_samples - len(malignant)]
    selected_samples = malignant + benign

    print(f"Elaborazione di {len(selected_samples)} casi BCN20000 Raw...")

    for idx, (path, label, img_id) in enumerate(selected_samples):
        label_str = "Melanoma" if label == 1 else "Benigno"
        rgb_img, input_tensor = preprocess_image(path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            p_cnn = torch.sigmoid(cnn_model(input_tensor)).item()
            p_hyb = torch.sigmoid(hybrid_model(input_tensor)).item()

        vis_cnn = show_cam_on_image(rgb_img, cam_cnn(input_tensor=input_tensor, targets=cam_targets)[0, :], use_rgb=True)
        vis_hyb = show_cam_on_image(rgb_img, cam_hybrid(input_tensor=input_tensor, targets=cam_targets)[0, :], use_rgb=True)

        esito_cnn = "Melanoma" if p_cnn >= args.threshold else "Benigno"
        esito_hyb = "Melanoma" if p_hyb >= args.threshold else "Benigno"

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Immagine Raw: {img_id}\nGround Truth: {label_str}", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        color_cnn = "green" if esito_cnn == label_str else "red"
        axes[1].imshow(vis_cnn)
        axes[1].set_title(f"ResNet-50 (CNN Raw)\nP(Melanoma): {p_cnn:.4f}\nEsito: {esito_cnn}", fontsize=11, color=color_cnn)
        axes[1].axis("off")

        color_hyb = "green" if esito_hyb == label_str else "red"
        axes[2].imshow(vis_hyb)
        axes[2].set_title(f"MobileViT-S (Hybrid Raw)\nP(Melanoma): {p_hyb:.4f}\nEsito: {esito_hyb}", fontsize=11, color=color_hyb)
        axes[2].axis("off")

        plt.tight_layout()
        out_file = os.path.join(args.output_dir, f"bcn_raw_gradcam_{idx+1}_{label_str}.png")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[{idx+1}/{len(selected_samples)}] Salvato: {out_file}")

    print(f"\n[BCN20000 XAI] Generazione terminata in {args.output_dir}/")

if __name__ == "__main__":
    main()
