# Fashion-AEye
Shopping for Clothes and Accessories reimagined using CLIP image embeddings and NLP Vibe Engine

# Flickd AI Hackathon Submission  
**Smart Tagging & Vibe Classification Engine**

This project presents a functional backend system that can intelligently process short fashion videos ("Reels") and generate structured data about the fashion items present and the overall aesthetic or "vibe" of the content.

The system operates on video inputs, detects clothing/accessories, matches them to a product catalog, classifies fashion aesthetics, and outputs structured `.json` results for each video.

---
## 🧩 Key Steps Undertaken

### 1. Frame Extraction
- Extracted 3–5 frames per video using OpenCV at a configurable interval.
- Saved as `/frames/video_id/frame_001.jpg`.

### 2. Object Detection with YOLOS (Fashionpedia)
- Used `valentinafeve/yolos-fashionpedia` from Hugging Face.
- Detected fashion items: tops, bags, dresses, etc.
- Cropped detected bounding boxes and stored for product matching.

### 3. Product Matching (CLIP + FAISS)
- Generated CLIP embeddings for:
  - Catalog products
  - Cropped detections
- Used FAISS index + cosine similarity.
- Labeled each match as:
  - `exact` (similarity > 0.9)
  - `similar` (0.75–0.9)
  - `no match` (< 0.75)

### 4. Vibe Classification
- Used Sentence-BERT (`all-MiniLM-L6-v2`) to embed:
  - Combined product metadata OR simulated captions.
- Compared with vector embeddings of supported vibes.
- Assigned top-3 vibes per video.

### 5.  Final Output
- Returned per-video JSON like:

```json
{
  "video_id": "reel_001",
  "vibes": ["Clean Girl", "Coquette", "Y2K"],
  "products": [
    {
      "type": "dress",
      "match_type": "similar",
      "matched_product_id": "SKU_12345",
      "confidence": 0.87
    }
  ]
}


