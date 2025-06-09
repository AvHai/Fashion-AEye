import os
import cv2
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import pipeline
import clip
from sentence_transformers import SentenceTransformer, util

from catalog_indexer import CatalogIndexer

# ----------------------------
# Paths & Constants
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CATALOG_PATH = "catalog.csv"
CATALOG_IMAGE_DIR = "catalog/images"
VIDEO_DIR = "videos"
FRAME_DIR = "frames"
CROP_DIR = "crops"
OUTPUT_DIR = "outputs"
INDEX_PATH = "faiss_catalog.index"
ID_PATH = "product_ids.pkl"

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Step 1: Load Models
# ----------------------------
print("🔁 Loading models...")
clip_model, preprocess_clip = clip.load("ViT-B/32", device=DEVICE)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
yolo_pipe = pipeline("object-detection", model="valentinafeve/yolos-fashionpedia")

vibes_list = ["Coquette", "Clean Girl", "Cottagecore", "Streetcore", "Y2K", "Boho", "Party Glam"]
vibe_embeddings = sentence_model.encode(vibes_list, convert_to_tensor=True)

# ----------------------------
# Step 2: Build or Load FAISS
# ----------------------------
print("📦 Preparing catalog index...")
catalog_df = pd.read_csv(CATALOG_PATH)
indexer = CatalogIndexer(clip_model, preprocess_clip, DEVICE)

if os.path.exists(INDEX_PATH):
    indexer.load(INDEX_PATH, ID_PATH)
else:
    indexer.build_from_folder(CATALOG_IMAGE_DIR, catalog_df["id"].tolist())
    indexer.save(INDEX_PATH, ID_PATH)

# ----------------------------
# Step 3: Process Videos
# ----------------------------
print("🎞️ Processing videos...")

for video_file in tqdm(os.listdir(VIDEO_DIR)):
    if not video_file.endswith(".mp4"):
        continue

    video_id = os.path.splitext(video_file)[0]
    video_path = os.path.join(VIDEO_DIR, video_file)
    frame_subdir = os.path.join(FRAME_DIR, video_id)
    crop_subdir = os.path.join(CROP_DIR, video_id)
    os.makedirs(frame_subdir, exist_ok=True)
    os.makedirs(crop_subdir, exist_ok=True)

    # Extract frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 2)
    count, frame_idx = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(frame_subdir, f"frame_{frame_idx:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        count += 1
    cap.release()

    # Detect + crop
    product_matches = []
    text_buffer = []

    for frame_file in sorted(os.listdir(frame_subdir)):
        frame_path = os.path.join(frame_subdir, frame_file)
        pil_img = Image.open(frame_path).convert("RGB")
        results = yolo_pipe(pil_img)

        for i, obj in enumerate(results):
            label = obj["label"]
            conf = obj["score"]
            box = obj["box"]
            x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
            crop = pil_img.crop((x1, y1, x2, y2))
            crop_name = f"{frame_file[:-4]}_{label}_{i}.jpg"
            crop_path = os.path.join(crop_subdir, crop_name)
            crop.save(crop_path)

            # Match
            match_result = indexer.match(crop, k=1)[0]
            sim = match_result["similarity"]
            match_type = "exact" if sim > 0.9 else "similar" if sim > 0.75 else "no match"

            if match_type != "no match":
                matched_id = match_result["product_id"]
                product_type = catalog_df[catalog_df["id"] == matched_id]["product_type"].values[0]
                product_matches.append({
                    "type": product_type,
                    "match_type": match_type,
                    "matched_product_id": matched_id,
                    "confidence": round(sim, 3)
                })

                # Text for vibe classification
                row = catalog_df[catalog_df["id"] == matched_id]
                combined = " ".join(row[["title", "description", "product_tags"]].fillna("").values[0])
                text_buffer.append(combined)

    # Vibe Classification
    combined_text = " ".join(text_buffer)
    text_embedding = sentence_model.encode(combined_text, convert_to_tensor=True)
    vibe_scores = util.cos_sim(text_embedding, vibe_embeddings)[0]
    top_indices = vibe_scores.topk(3).indices.tolist()
    top_vibes = [vibes_list[i] for i in top_indices]

    # Final Output
    final_output = {
        "video_id": video_id,
        "vibes": top_vibes,
        "products": product_matches
    }

    with open(os.path.join(OUTPUT_DIR, f"{video_id}.json"), "w") as f:
        json.dump(final_output, f, indent=2)

print("✅ All videos processed. Check outputs/ for results.")
