# Twitter Sentiment Classification with CNN + LSTM + GloVe

This project performs sentiment classification on a dataset of tweets using deep learning techniques. It combines convolutional layers (CNN), a Bidirectional LSTM network and pretrained GloVe embeddings to classify tweets into four sentiment categories: Positive, Negative, Neutral and Irrelevant.

## Dataset

The dataset (`twitter_data.csv`) contains tweets labeled with four sentiment categories. Each tweet is associated with a user and a sentiment label. Missing values are removed during preprocessing.

## Features

### Text Preprocessing
The raw tweet text goes through several cleaning steps before being fed into the model. Links, mentions, hashtags and punctuation are removed, all text is lowercased, and each word is lemmatized using WordNet to reduce words to their base form. This ensures the model sees cleaner and more consistent input.

### Tokenization & Padding
The cleaned tweets are tokenized using the top 10000 most frequent words in the training set. Each sequence is padded to a fixed length of 64 tokens to ensure uniform input size across all samples.

### GloVe Embeddings
Instead of learning word representations from scratch, the model uses pretrained GloVe embeddings (100-dimensional) trained on billions of words. This gives the model a head start in understanding word meaning and relationships.

### Model Architecture
The model is built as follows:
- **Embedding layer** initialized with GloVe weights
- **Two Conv1D layers** with BatchNormalization and Dropout to extract local patterns from the text
- **Bidirectional LSTM** to capture context from both directions in the sequence
- **GlobalMaxPooling1D** to extract the most important features
- **Dense layers** with Dropout to reduce overfitting and produce the final prediction

### Training
The model is trained with EarlyStopping to prevent overfitting and ReduceLROnPlateau to automatically lower the learning rate when the validation loss stops improving.

## Model Performance

The model achieves 80% accuracy on the test set. 

Across all four sentiment classes the model performs consistently, with Negative sentiment being the strongest at 0.85 F1-score, followed by Positive at 0.81, Neutral at 0.76, and Irrelevant at 0.75 which is expected as it is the hardest class to distinguish from the others.
