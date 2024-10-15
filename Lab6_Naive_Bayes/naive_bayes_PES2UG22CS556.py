import numpy as np
import warnings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore", category=RuntimeWarning)


class NaiveBayesClassifier:
    @staticmethod
    def preprocess(sentences, categories):
        cleaned_sentences = []
        cleaned_categories = []

        for sentence, category in zip(sentences, categories):
            if category is None or category == "wrong_label":
                continue  
            
            words = [word for word in sentence.lower().split() if word not in ENGLISH_STOP_WORDS]
            cleaned_sentence = ' '.join(words)
            
            cleaned_sentences.append(cleaned_sentence)
            cleaned_categories.append(category)

        return cleaned_sentences, cleaned_categories

    @staticmethod
    def fit(X, y):
        n_docs, n_words = X.shape
        class_probs = {}
        word_probs = {}

        classes, class_counts = np.unique(y, return_counts=True)
        for cls, count in zip(classes, class_counts):
            class_probs[cls] = count / n_docs

        for cls in classes:
            class_docs = X[np.array(y) == cls]  # Documents in the class
            word_count_in_class = class_docs.sum(axis=0) + 1  # Add-1 smoothing
            total_word_count = word_count_in_class.sum()
            
            word_probs[cls] = word_count_in_class / total_word_count

        return class_probs, word_probs

    @staticmethod
    def predict(X, class_probs, word_probs, classes):
        predictions = []

        for x in X:
            class_scores = {}
            for cls in classes:

                class_score = np.log(class_probs[cls])


                word_likelihoods = word_probs[cls]
                class_score += np.sum(x * np.log(word_likelihoods))

                class_scores[cls] = class_score

            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)

        return predictions
