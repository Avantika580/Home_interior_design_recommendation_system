# Categorized Interior Design Style Recommendation System

## Problem Statement

Selecting an interior design style that matches user preferences can be challenging due to the diversity of visual aesthetics and overlapping characteristics across categories.

This project builds a structured classification and recommendation system that organizes interior design images into defined categories and enables efficient style-based filtering.


## Dataset Overview

- **Total Images:** 420  
- **Categories:** 6  
  - Modern  
  - Traditional  
  - Farmhouse  
  - Minimalism  
  - Boho  
  - Industrial  
- **Images per Category:** 70  

The dataset was curated and categorized based on defining architectural and stylistic characteristics of each interior design style.


## System Architecture
User Interaction (Streamlit UI)
↓
Category Selection / Preference Input
↓
Style Filtering Logic
↓
Image Retrieval from Structured Dataset
↓
Recommendation Display


### Design Considerations

- Modular architecture separating UI and filtering logic  
- Pre-structured dataset for constant-time category lookup  
- Lightweight in-memory processing  
- Fast rendering using Streamlit  


## Key Features

- Interactive Streamlit-based interface  
- Categorized image visualization  
- Structured style-based filtering  
- Optimized for low-latency response  
- Clean and extensible dataset design  


## Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  

## Performance Characteristics

- Supports 6 style categories with balanced dataset distribution  
- Handles 420+ images efficiently  
- Sub-second response time during local inference  
- Designed for scalable category expansion

## How to Run Locally

```bash
git clone https://github.com/Avantika580/Home_interior_design_recommendation_system.git
cd Home_interior_design_recommendation_system
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Future Enhancements

- Add ML-based image similarity scoring  
- Implement embedding-based recommendation  
- Add user session memory  
- Expand dataset to 1000+ images  
- Introduce backend API layer (FastAPI)  

---

## 👩‍💻 Author

**Avantika Gurav**  
B.Tech Information Technology  
Pune, India  
