import sys
from transformers import pipeline

def run_sa():
    user_input = input("Enter text for sentiment analysis (default: 'I love coding in python!'): ")
    text = user_input if user_input else "I love coding in python!"
    sentiment_analysis(text)

def sentiment_analysis(text):
    classifier = pipeline('sentiment-analysis')
    result = classifier(text)[0]
    print(f"The text \"{text}\" was classified as {result['label']} with a score of {round(result['score'], 4) * 100}%")
