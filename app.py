import streamlit as st
import pickle
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix

nltk.download("punkt")
nltk.download("stopwords")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = nltk.word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]

    return " ".join(words)

# ---------------- UI ----------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detection System")

news = st.text_area("Enter News Text Here")

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter news text")
    else:
        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()

        if prediction == 1:
            st.success(f"REAL NEWS ✅ (Confidence: {confidence*100:.2f}%)")
        else:
            st.error(f"FAKE NEWS ❌ (Confidence: {confidence*100:.2f}%)")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📊 Confusion Matrix")

if st.button("Show Confusion Matrix"):

    st.info("Make sure you load X_test and y_test if you want real evaluation")

    # TEMP DEMO MATRIX (replace with real values)
    cm = [[50, 8],
          [4, 38]]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)