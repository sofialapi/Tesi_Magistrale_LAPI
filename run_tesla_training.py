import os
import argparse
import pandas as pd
import torch

from src.config import RAW_ISIC_DIR, PROCESSED_ISIC_DIR
from src.training.train import run_stratified_kfold

def main():
    parser = argparse.ArgumentParser(description="Addestramento Modelli Tesi su Server Tesla")
    parser.add_argument("--case", type=str, required=True, 
                        choices=["cnn_only", "hybrid_only", "cnn_multimodal", "hybrid_multimodal"],
                        help="Caso di studio da addestrare")
    parser.add_argument("--gpu", type=int, default=0, help="ID della GPU Tesla P100 da utilizzare")
    parser.add_argument("--epochs", type=int, default=30, help="Numero di epoche per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimensione del batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate per AdamW")
    
    args = parser.parse_args()

    #VECCHIO
    # Imposta la GPU specifica da allocare sul server
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #print(f"[TESLA EXEC] Device allocato: {device} (Physical GPU ID: {args.gpu})")
    
    #NUOVO
    # Slurm gestisce in automatico CUDA_VISIBLE_DEVICES isolando la P100 allocata
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TESLA EXEC] Device allocato: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    #csv_path = os.path.join(RAW_ISIC_DIR, "metadata_sample.csv")
    #sotto: nuovo path per cartella locale tesla
    csv_path = os.path.join(RAW_ISIC_DIR, "metadata_isic_subset.csv")
    df = pd.read_csv(csv_path)
    
    run_stratified_kfold(
        df_metadata=df,
        image_dir=PROCESSED_ISIC_DIR,
        case_study=args.case,
        k_folds=5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

if __name__ == "__main__":
    main()