import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os
import pickle
from PIL import Image
import random
import numpy as np

# ================================
# ⚙️ Dataset & Cache
# ================================
DATASET_DIR = r"C:\Users\Avantika\Downloads\Categorized-Interior-Design-Style-main"
CACHE_FILE = os.path.join(DATASET_DIR, "fa_cache.pkl")

@st.cache_resource
def load_cache():
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)

# ================================
# Load cache
# ================================
cache_data = load_cache()
labels = cache_data.get("labels", [])
apr_rules = cache_data.get("apr_rules", [])
style_names = cache_data.get("style_names", labels)

# Deduplicate styles for multiselect
unique_styles_map = {}
for s in style_names:
    key = s.strip().lower().replace(" ", "").replace("-", "")
    if key not in unique_styles_map:
        unique_styles_map[key] = s

all_styles = sorted(unique_styles_map.values())

# ================================
# Helper Functions
# ================================
def normalize(name):
    return name.replace(" ", "").replace("-", "").replace("_", "").lower()

def id_to_style(idx):
    try:
        return style_names[int(idx)]
    except:
        return str(idx)

@st.cache_data
def get_images_by_style(style, max_samples=4):
    style_norm = normalize(style)
    style_folder = None
    for f in os.listdir(DATASET_DIR):
        if os.path.isdir(os.path.join(DATASET_DIR, f)) and normalize(f) == style_norm:
            style_folder = os.path.join(DATASET_DIR, f)
            break
    if not style_folder:
        return []
    imgs = [os.path.join(style_folder, f)
            for f in os.listdir(style_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return random.sample(imgs, min(len(imgs), max_samples))

# ================================
# Recommendation System
# ================================
@st.cache_data
def get_recommendations(selected_styles):
    selected_norm = {normalize(s) for s in selected_styles}
    recommended = []

    for rule in apr_rules:
        antecedents = {normalize(id_to_style(a)) for a in rule.get("antecedents", [])}
        consequents = {normalize(id_to_style(c)) for c in rule.get("consequents", [])}
        support = float(rule.get("support", 0))
        confidence = float(rule.get("confidence", 0))

        if antecedents & selected_norm:
            for c in consequents - selected_norm:
                proper_name = unique_styles_map.get(c, c)
                recommended.append({
                    "style": proper_name,
                    "support": support,
                    "confidence": confidence
                })

    final = {}
    for r in recommended:
        if r["style"] not in final or r["confidence"] > final[r["style"]]["confidence"]:
            final[r["style"]] = r

    return list(final.values())

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Interior Design Style Recommender", layout="wide")
st.title("Interior Design Style Recommender")

selected_styles = st.multiselect("Select one or more styles:", all_styles)

if selected_styles:
    st.subheader("Selected Style Samples")
    for style in selected_styles:
        imgs = get_images_by_style(style)
        if imgs:
            cols = st.columns(len(imgs))
            for col, img_path in zip(cols, imgs):
                col.image(Image.open(img_path), caption=style, use_container_width=True)

    recommendations = get_recommendations(selected_styles)

    if recommendations:
        st.subheader("Recommended Co-occurring Styles")
        for rec in recommendations:
            style = rec["style"]
            st.markdown(f"**{style}** — Support: {rec['support']:.2f}, Confidence: {rec['confidence']:.2f}")

            imgs = get_images_by_style(style)
            if imgs:
                cols = st.columns(len(imgs))
                for col, img_path in zip(cols, imgs):
                    col.image(Image.open(img_path), caption=style, use_container_width=True)
    else:
        st.info("No recommendations found for selected styles.")
else:
    st.info("Select one or more styles to see recommendations.")
