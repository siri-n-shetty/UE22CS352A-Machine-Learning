from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from naive_bayes_PES2UG22CS556 import NaiveBayesClassifier
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_tests(test_cases):
    # Defining the training sentences and categories
    sentences = [
        "The new smartphone has a stunning display and battery life.",  # Technology
        "Traveling to Japan was a dream come true.",  # Travel
        "The latest movie has been a box office hit.",  # Entertainment
        "AI is transforming the future of work.",  # Technology
        "Exploring the ancient ruins was a memorable experience.",  # Travel
        "The concert was an unforgettable experience.",  # Entertainment
        "5G networks are rolling out across the globe.",  # Technology
        "Hiking in the Alps offers breathtaking views.",  # Travel
        "The streaming platform has released new shows.",  # Entertainment
        "Electric cars are the future of transportation.",  # Technology
        "Traveling to tropical islands is a fantastic getaway.",  # Travel
        "Missing label",  # Missing label (noise)
        "Virtual reality is reshaping gaming.",  # Technology
        "Incorrect label"  # Incorrect label (noise)
    ]

    categories = [
        "technology", "travel", "entertainment", "technology", "travel", "entertainment", 
        "technology", "travel", "entertainment", "technology", "travel", None, 
        "technology", "wrong_label"
    ]

    # Preprocessing step
    sentences, categories = NaiveBayesClassifier.preprocess(sentences, categories)

    # Vectorizing the text data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(sentences)

    # Fitting the Naive Bayes model
    class_probs, word_probs = NaiveBayesClassifier.fit(X_train_vec.toarray(), categories)

    num_passed = 0

    for test_sentence, correct_category in test_cases:
        test_vector = vectorizer.transform([test_sentence]).toarray()
        prediction = NaiveBayesClassifier.predict(test_vector, class_probs, word_probs, np.unique(categories))[0]

        if prediction == correct_category:
            print(f"Test Passed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")
            num_passed += 1
        else:
            print(f"Test Failed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")

    return num_passed


if __name__ == "__main__":

    test_cases = [
        ("The smartphone has amazing battery life.", "technology"),
        ("Exploring the ancient temples in Angkor Wat is a breathtaking experience.", "travel"),
        ("The movie was a box office hit.", "entertainment"),
        ("AI will shape the future of transportation.", "technology"),
        ("Traveling through Europe has been a life-changing experience.", "travel"),
        ("The concert was mind-blowing.", "entertainment"),
        ("The latest advancements in electric cars are impressive.", "technology"),
        ("A road trip across the country offers countless adventures.", "travel"),
        ("The latest blockbuster movie kept everyone on the edge of their seats.", "entertainment"),
        ("The impact of technology on our daily lives is profound.", "technology")
    ]

    num_passed = run_tests(test_cases)
    print(f"\nNumber of Test Cases Passed: {num_passed} out of {len(test_cases)}")
