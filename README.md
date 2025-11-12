** Research Project – 2025 **

A hybrid deep learning model combining Sentence-BERT (SBERT) and BiLSTM architectures for sentiment and depression detection in social media text.

** Overview **

This research project explores the use of hybrid deep learning models to analyze linguistic markers in social media posts for identifying signs of depression and emotional distress.

The model integrates Sentence-BERT embeddings for contextual understanding of text with a BiLSTM network for sequential learning.

The primary goal of this research is to improve explainability and accuracy in mental health text classification models by leveraging transfer learning, deep contextual embeddings, and interpretable linguistic features.

** Objectives **

Build an interpretable deep learning model for detecting linguistic indicators of depression.

Compare performance between baseline models (SBERT, BiLSTM) and the proposed hybrid approach.

Evaluate linguistic feature contributions to model explainability.

Support ongoing research in AI for Mental Health and Explainable NLP.

** Model Architecture **
Input Text → Preprocessing → SBERT Embeddings → BiLSTM → Dense Layers → Output (Classification)


SBERT: Extracts sentence-level contextual embeddings.

BiLSTM: Captures sequential dependencies and emotional tone.

Dense Layers: Perform classification (Depressed / Not Depressed).

** Technologies & Libraries **

Python 3.10+

PyTorch

Transformers (Hugging Face)

Pandas, NumPy, Matplotlib

Scikit-learn

TextBlob / NLTK / SpaCy

Google Colab / Jupyter Notebook

** Dataset **

Publicly available social media text datasets (e.g., Reddit or Twitter-based depression detection datasets).

Data preprocessing includes:

Tokenization

Stopword removal

Lemmatization

Text normalization

⚠️ Note: Dataset links are omitted for ethical and privacy reasons. Use publicly available research datasets for replication.

Results are based on experimental datasets and may vary depending on preprocessing and hyperparameters.

** Explainability **

To ensure interpretability, the model includes:

Linguistic feature extraction (e.g., pronoun ratios, sentiment scores, readability indices).

Attention visualization to highlight key words influencing classification.

Feature importance analysis using SHAP/LIME.

** Author **

Kgodiso Austin Leboho

kgodisoaustinleboho@gmail.com

Midrand, South Africa
