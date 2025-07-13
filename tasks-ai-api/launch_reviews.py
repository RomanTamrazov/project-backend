import joblib
clf = joblib.load("logreg_sentiment_model.joblib")
vec = joblib.load('tfidf_vectorizer_to_lounch.joblib')
labels_map = {
    0: "🌞",
    1: "😐",
    2: "🌩️",
    3: "🤨"
}

def predict_sentiment(text):
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)[0]
    return labels_map[pred]

print(predict_sentiment("телефон ужасный"))
