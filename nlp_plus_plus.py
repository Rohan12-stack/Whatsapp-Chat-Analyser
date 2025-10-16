# nlp_plus_plus.py

# 1. Imports
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load and preprocess WhatsApp chat data (simplified)
def load_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    messages = []
    for line in data:
        match = re.match(r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2})\s-\s(.*?):\s(.*)", line)
        if match:
            date, time, sender, message = match.groups()
            messages.append([date, time, sender, message])

    df = pd.DataFrame(messages, columns=['date', 'time', 'sender', 'message'])
    return df

# 3. Rule-Based Sentiment Analysis using VADER
def apply_vader_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    def get_sentiment(msg):
        score = analyzer.polarity_scores(msg)['compound']
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['Rule_Based_Sentiment'] = df['message'].apply(get_sentiment)
    return df

# 4. ML-Based Sentiment Classification
def apply_ml_sentiment(df):
    # Use rule-based sentiment as pseudo-labels
    df = df[df['Rule_Based_Sentiment'] != '']
    X = df['message']
    y = df['Rule_Based_Sentiment']

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df['ML_Based_Sentiment'] = model.predict(X_vec)

    # Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return df

# 5. Visualization (Optional)
def plot_sentiment_distribution(df):
    plt.figure(figsize=(10,5))
    sns.countplot(x='Rule_Based_Sentiment', data=df, palette='Set2')
    plt.title("Sentiment Distribution (Rule-Based)")
    plt.show()

    plt.figure(figsize=(10,5))
    sns.countplot(x='ML_Based_Sentiment', data=df, palette='Set1')
    plt.title("Sentiment Distribution (ML-Based)")
    plt.show()

# 6. Main Driver Function
if __name__ == "__main__":
    # Change file path as needed
    chat_file = "WhatsApp Chat with Facial Emotion Recognition!.txt"
    df = load_chat(chat_file)

    print("\nApplying Rule-Based Sentiment...")
    df = apply_vader_sentiment(df)

    print("\nApplying ML-Based Sentiment...")
    df = apply_ml_sentiment(df)

    print("\nVisualizing Sentiment Distributions...")
    plot_sentiment_distribution(df)

    # Save results
    df.to_csv("sentiment_comparison_output.csv", index=False)
    print("\nSentiment analysis complete. Results saved to sentiment_comparison_output.csv")
