import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------- Fix KMeans multi-thread issue on Windows ----------------
os.environ["OMP_NUM_THREADS"] = "1"

# ================================
# 1️⃣ Dataset Path
# ================================
DATASET_DIR = r"C:\Users\Avantika\Downloads\Categorized-Interior-Design-Style-main"

# ================================
# 2️⃣ Parameters
# ================================
NUM_CLUSTERS = 5
VALID_EXTS = [".jpg", ".jpeg", ".png"]

# ================================
# 3️⃣ Feature extraction
# ================================
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_feature(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = vgg_model.predict(x, verbose=0)
    return feat.flatten()

# ================================
# 4️⃣ Prepare dataset
# ================================
def prepare_dataset(force_rebuild=False):
    if os.path.exists("fa_cache.pkl") and not force_rebuild:
        print("[INFO] Loading cached features and rules...")
        with open("fa_cache.pkl", "rb") as f:
            return pickle.load(f)

    print("[INFO] Extracting features...")
    features = []
    labels = []



    categories = sorted(os.listdir(DATASET_DIR))
    for category in categories:
        cat_path = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(cat_path):
            continue
        print(f"[INFO] Processing category: {category}")
        for img_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_name)
            if not any(img_name.lower().endswith(ext) for ext in VALID_EXTS):
                continue
            try:
                feat = extract_feature(img_path)
                features.append(feat)
                labels.append(category)
            except Exception as e:
                print(f"[WARNING] Could not process image: {img_path}, error: {e}")

    features = np.array(features)
    labels = np.array(labels)

       
       # 🔍 DEBUG: Check your labels
    print("Unique labels detected:", sorted(set(labels)))
    print("Total distinct styles:", len(set(labels)))
    print("Total images:", len(labels))

    if len(features) == 0:
        raise ValueError("[ERROR] No valid images found! Check your dataset folder.")

    # Adjust clusters if fewer images than NUM_CLUSTERS
    n_clusters = min(NUM_CLUSTERS, len(features))

    # ================================
    # 5️⃣ Clustering
    # ================================
    print("[INFO] Clustering features with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)

    # Create transaction dataframe for Apriori
    df_trans = pd.DataFrame({"style": labels, "cluster": clusters})
    basket = pd.get_dummies(df_trans, columns=["style"], prefix="", prefix_sep="").groupby("cluster").max()

    # ================================
    # 6️⃣ Apriori
    # ================================
    print("[INFO] Running Apriori...")
    apr_df = apriori(basket, min_support=0.01, use_colnames=True)
    apr_df = association_rules(apr_df, metric="lift", min_threshold=1)

    # Convert frozensets to normal sets and store support/confidence as float
    apr_rules = []
    for _, row in apr_df.iterrows():
        apr_rules.append({
            "antecedents": set(row['antecedents']),
            "consequents": set(row['consequents']),
            "support": float(row['support']),
            "confidence": float(row['confidence'])
        })
    print(f"[INFO] Apriori found {len(apr_rules)} rules.")

    # ================================
    # 7️⃣ Save cache
    # ================================
    cache_data = {
        "features": features,
        "labels": labels,
        "clusters": clusters,
        "apr_rules": apr_rules
    }
    with open("fa_cache.pkl", "wb") as f:
        pickle.dump(cache_data, f)
    print("[INFO] Model + rules saved to fa_cache.pkl")
    return cache_data

# ---------------- CLI entry ----------------
if __name__ == "__main__":
    print("[INFO] Preparing dataset and building apriori rules (may take time)...")
    prepare_dataset(force_rebuild=False)
    print("[INFO] Done.")
