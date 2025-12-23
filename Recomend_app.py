import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    df = pd.read_csv(r"C:\Users\ISMAIMZ\OneDrive - Hapag-Lloyd AG\Documents\Books\Projects\Recommendation system\archive\clean_data.csv")  
    df = df.drop(columns=["Unnamed: 0", "ImageURL"])
    for col in ["Name","Brand","Category","Description","Tags"]:
        df[col] = df[col].fillna("").astype(str)
 
    def clean(s):
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    for col in ["Name","Brand","Category","Description","Tags"]:
        df[col] = df[col].map(clean)
    df["doc"] = (df["Name"] + " " + df["Brand"] + " " + df["Category"] +
                 " " + df["Description"] + " " + df["Tags"])
   
    rating = (df["Rating"] - df["Rating"].min()) / (df["Rating"].max() - df["Rating"].min() + 1e-9)
    reviews = np.log1p(df["ReviewCount"])
    reviews = (reviews - reviews.min()) / (reviews.max() - reviews.min() + 1e-9)
    df["pop_score"] = 0.6 * rating + 0.4 * reviews
    return df

df = load_data()



def build_model(df):
    tfidf = TfidfVectorizer(min_df=3, ngram_range=(1,2), stop_words="english", max_features=20000)
    X = tfidf.fit_transform(df["doc"])
    sim = cosine_similarity(X, X)
    id_to_idx = {pid:i for i,pid in enumerate(df["ProdID"])}
    idx_to_id = {i:pid for pid,i in id_to_idx.items()}
    return X, sim, id_to_idx, idx_to_id

X, sim, id_to_idx, idx_to_id = build_model(df)


def similar_items(prod_id, k=10, w_pop=0.2):
    if prod_id not in id_to_idx:
        return pd.DataFrame()
    i = id_to_idx[prod_id]
    scores = (1 - w_pop) * sim[i] + w_pop * df["pop_score"].values
    scores[i] = -np.inf
    top_idx = np.argsort(-scores)[:k]
    return df.iloc[top_idx][["ProdID","Name","Brand","Category","Rating","ReviewCount"]]

def recommend_from_history(prod_ids, k=10, w_pop=0.2):
    idxs = [id_to_idx[p] for p in prod_ids if p in id_to_idx]
    if not idxs:
        return df.sort_values("pop_score", ascending=False).head(k)[["ProdID","Name","Brand","Category","Rating","ReviewCount"]]
    profile = X[idxs].mean(axis=0)
    scores = cosine_similarity(profile, X).ravel()
    scores = (1 - w_pop) * scores + w_pop * df["pop_score"].values
    for i in idxs: scores[i] = -np.inf
    top_idx = np.argsort(-scores)[:k]
    return df.iloc[top_idx][["ProdID","Name","Brand","Category","Rating","ReviewCount"]]

def diversified_top_picks(k=10):
    cand = df.copy()
    cand["score"] = cand["pop_score"]
    top, seen_cat = [], set()
    for _, row in cand.sort_values("score", ascending=False).iterrows():
        cat = row["Category"].split(",")[0] if row["Category"] else "misc"
        if cat not in seen_cat or len(top) < k//2:
            top.append(row)
            seen_cat.add(cat)
        if len(top) >= k: break
    return pd.DataFrame(top)[["ProdID","Name","Brand","Category","Rating","ReviewCount"]]


st.title("🛒 E-Commerce Recommendation System")

option = st.sidebar.selectbox("Choose scenario:", 
                              ["Similar Items", "Recommendations from History", "Cold-Start Top Picks"])

if option == "Similar Items":
    prod_id = st.text_input("Enter a Product ID:")
    if prod_id:
        recs = similar_items(prod_id, k=10)
        st.write("### Recommended Similar Items")
        st.dataframe(recs)

elif option == "Recommendations from History":
    prod_ids = st.text_input("Enter Product IDs (comma separated):")
    if prod_ids:
        prod_list = [p.strip() for p in prod_ids.split(",")]
        recs = recommend_from_history(prod_list, k=10)
        st.write("### Recommendations Based on History")
        st.dataframe(recs)

elif option == "Cold-Start Top Picks":
    recs = diversified_top_picks(k=10)
    st.write("### Diversified Popular Picks")
    st.dataframe(recs)
