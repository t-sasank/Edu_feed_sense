# Edu_feed_sense
Edu Feed Sense (Educational feedback Sentiment analyzer)– An AI-powered system for analyzing student feedback on faculty using NLP &amp; ML models. It extracts sentiments, builds consensus, and generates actionable insights to improve faculty performance.


Overall structure of the files would be :

project/
│
├── app.py                # Main application file
├── templates/            # Create a new folder templates
│   └── index.html        # HTML template
|   └── result.html       # Result HTML page
├── uploads               #Folder with the csv Feedback files
├── Analyzed_feedback.json  # Result of the Feedback
├── faculty.json            # Faculty List
├── model.py                # Models for Sentiment Analysis



🚀 Features

Collects and processes student feedback.

Performs sentiment analysis (positive, negative, neutral).

Uses multiple ML/NLP models (Naive Bayes, BERT, RoBERTa, GPT, XLNet, ALBERT, T5, ERNIE, XLM-R).

Generates faculty performance insights based on feedback.

Helps in decision-making with data-driven reports.

🛠️ Tech Stack

Languages: Python

Libraries/Frameworks: scikit-learn, pandas, numpy, matplotlib, transformers, nltk, torch

Models: Multinomial Naive Bayes, BERT-family models, GPT-2, XLNet, ALBERT, T5, ERNIE, XLM-R

📂 Project Workflow

Data Collection – Gather faculty feedback data.

Preprocessing – Clean and prepare text (stopwords removal, tokenization, lemmatization).

Model Training – Train and fine-tune ML/NLP models.

Sentiment Analysis – Predict sentiment of each feedback.

Consensus Building – Combine results to provide overall faculty insights.

Visualization – Graphs and charts for easy understanding.

📊 Example Output

Sentiment distribution (Positive %, Negative %, Neutral %).

Word clouds of student opinions.

Comparative performance analysis of faculty.
