# 🛒 E-Commerce Recommendation System

## Overview
This project implements a **content-based recommendation engine** for e-commerce products using Kaggle dataset metadata (Name, Brand, Category, Description, Tags, Ratings, ReviewCount).  
It suggests similar items, generates recommendations from user history, and provides cold-start diversified popular picks. Developed a hybrid e-commerce recommendation system using TF‑IDF and popularity-based reranking to suggest similar products from catalog metadata.

## Features
- **Content-based filtering** with TF‑IDF vectorization and cosine similarity.  
- **Popularity-aware reranking** using ratings and review counts.  
- **User history aggregation** to recommend items based on multiple liked products.  
- **Cold-start recommendations** with category diversification.  
- **Interactive Streamlit app** for real-time product search and recommendations.  

## Dataset
- Kaggle e-commerce dataset containing product metadata:  
  - `ProdID`, `Name`, `Brand`, `Category`, `Description`, `Tags`, `Rating`, `ReviewCount`, `ImageURL`.

## Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn)  
- **Streamlit** for interactive UI  
- **TF‑IDF & Cosine Similarity** for text-based recommendations  

   git clone https://github.com/yourusername/ecommerce-recommender.git
   cd ecommerce-recommender
