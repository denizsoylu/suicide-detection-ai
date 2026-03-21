import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    """
    Temizlenmiş veriyi data/processed klasöründen yükler
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)

    data_path = os.path.join(
        base_dir, 'data', 'processed', 'cleaned_suicide_data.csv'
    )

    print(f"Veri şu adresten yükleniyor: {data_path}")
    return pd.read_csv(data_path)


def prepare_features(df):
    """
    Metinleri TF-IDF vektörlerine çevirir
    """
    X = df['text']
    y = df['class']  # suicide / non-suicide

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_tfidf = vectorizer.fit_transform(X)
    return X_tfidf, y


def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n=== Logistic Regression ===")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def train_naive_bayes(X_train, X_test, y_train, y_test):
    print("\n=== Naive Bayes ===")

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def main():
    df = load_data()
    print("TEXT NaN sayısı:", df['text'].isna().sum())
    print("CLASS NaN sayısı:", df['class'].isna().sum())
    print("Boş string sayısı:", (df['text'].str.strip() == "").sum())

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_logistic_regression(X_train, X_test, y_train, y_test)
    train_naive_bayes(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
