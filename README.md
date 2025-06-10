# Twitter Sentiment Classification with CNN + LSTM

This project performs sentiment classification on a dataset of tweets using deep learning techniques. It combines convolutional layers (CNN) and a Long Short-Term Memory (LSTM) network to classify tweets into four sentiment categories.

## Features

- Preprocessing and cleaning of tweet text (removal of links, mentions, hashtags, punctuation).
- Tokenization using the top 3500 most frequent tokens and padding of sequences.
- Neural network model includes:
  - Embedding layer
  - Convolutional layers (Conv1D) for feature extraction
  - LSTM layer for sequence modeling
- Categorical cross-entropy loss function.
- Evaluation with accuracy, precision, recall, and F1 score.
- Visualization of training loss over epochs.

## Dataset

The dataset (`twitter_data.csv`) contains tweets labeled with four sentiment categories. Each tweet is associated with a user and a sentiment label. Missing values are removed during preprocessing.

## Model Performance

The model achieves approximately **74% accuracy** on the test set, with the following evaluation metrics:
- Precision
- Recall
- F1-score
- Support
