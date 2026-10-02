import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def dull_razor(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    clean_img = cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return clean_img

def standardize_image(img_array: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    return denoised

def process_single_image(args):
    in_path, out_path = args
    if os.path.exists(out_path):
        return
    try:
        img = cv2.imread(in_path)
        if img is None:
            return
        clean = dull_razor(img)
        final_img = standardize_image(clean, target_size=(224, 224))
        cv2.imwrite(out_path, final_img)
    except Exception as e:
        print(f"Errore su {in_path}: {e}")

def main():
    input_dir = "data/ham10000/images"
    output_dir = "data/ham10000/processed_images"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Trovate {len(files)} immagini. Inizio preprocessing multithread (Dull Razor + Gaussiano)...")

    tasks = [(os.path.join(input_dir, f), os.path.join(output_dir, f)) for f in files]
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))

    print(f"Preprocessing terminato. Immagini elaborate salvate in: {output_dir}")

if __name__ == '__main__':
    main()

