"""
sentiment_analysis.py
---------------------
Core sentiment analysis module using NLTK's VADER (Valence Aware Dictionary
and sEntiment Reasoner).

VADER is a rule-based / lexicon-based sentiment analysis tool specifically
attuned to sentiments expressed in social media and product reviews.
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is downloaded once
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Initialize the VADER sentiment intensity analyzer
_analyzer = SentimentIntensityAnalyzer()

# Enhance lexicon with a few common e-commerce review terms
_CUSTOM_REVIEW_LEXICON = {
    "incredible": 2.8,
    "incredibly": 2.3,
    "durable": 2.0,
    "sturdy": 2.1,
    "flimsy": -2.2,
    "scratchy": -1.9,
    "unresponsive": -2.1,
    "overheats": -2.3,
    "overheating": -2.3,
}
_analyzer.lexicon.update(_CUSTOM_REVIEW_LEXICON)


def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    Analyzes the sentiment of a given review text string.

    Parameters:
        text (str): The customer review text.

    Returns:
        tuple[str, float]: (sentiment_label, compound_score)
            - sentiment_label: 'Positive', 'Negative', or 'Neutral'
            - compound_score: Normalized score ranging from -1.0 to +1.0
    """
    # Handle empty, None, or non-string inputs gracefully
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0.0

    # Obtain polarity scores dictionary: {'neg': float, 'neu': float, 'pos': float, 'compound': float}
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    # Standard VADER classification thresholds
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, round(compound, 3)


def process_reviews_dataframe(df: pd.DataFrame, text_column: str = "Review") -> pd.DataFrame:
    """
    Takes a DataFrame containing customer reviews, performs sentiment analysis
    on the specified column, and appends 'Sentiment' and 'Sentiment_Score' columns.

    Parameters:
        df (pd.DataFrame): Input dataframe containing customer reviews.
        text_column (str): Name of the column containing the review text. Default is 'Review'.

    Returns:
        pd.DataFrame: A copy of the dataframe with added sentiment columns.
    """
    # Work on a copy to avoid modifying original dataframe in-place
    processed_df = df.copy()

    # Case-insensitive column matching fallback if 'Review' not directly found
    matched_col = None
    for col in processed_df.columns:
        if col.strip().lower() == text_column.strip().lower():
            matched_col = col
            break

    if not matched_col:
        # Fallback: look for common alternatives like 'feedback', 'text', 'comment'
        alternatives = ["review", "feedback", "comment", "text", "description"]
        for col in processed_df.columns:
            if col.strip().lower() in alternatives:
                matched_col = col
                break

    if not matched_col:
        raise ValueError(
            f"Review column '{text_column}' not found. Available columns: {list(processed_df.columns)}"
        )

    # Apply sentiment analysis to each row
    sentiments = []
    scores = []

    for text in processed_df[matched_col]:
        label, score = analyze_sentiment(text)
        sentiments.append(label)
        scores.append(score)

    # Add the results as new columns
    processed_df["Sentiment"] = sentiments
    processed_df["Sentiment_Score"] = scores

    return processed_df
