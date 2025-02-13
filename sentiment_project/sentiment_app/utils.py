import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Paths to your model and vectorizer
path1 = 'C:/Users/aruni/OneDrive/Desktop/Django/sentiment_project/sentiment_app/models/sentiment_model.pkl'
path2 = 'C:/Users/aruni/OneDrive/Desktop/Django/sentiment_project/sentiment_app/models/vectorizer.pkl'

# Load the model and vectorizer
with open(path1, 'rb') as model_file:
    model = pickle.load(model_file)

with open(path2, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):  # Handle missing or non-string values
        return ""
    text = text.lower()  # Lowercase
    words = word_tokenize(text)  # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# Sentiment prediction function for Django
def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([preprocessed_review]).toarray()
    probabilities = model.predict_proba(vectorized_review)[0]  # Get probabilities for each class
    sentiment = model.predict(vectorized_review)[0]  # Get predicted class
    positive_prob = probabilities[1]  # Probability of being positive
    negative_prob = probabilities[0]  # Probability of being negative
    
    # Adjust the decision rule for 50% positive
    sentiment_result = "Positive" if positive_prob >= 0.50 else "Negative"

    return {
        "sentiment": sentiment_result,
        "positive_probability": f"{positive_prob:.2f}",  # Format for better display
        "negative_probability": f"{negative_prob:.2f}"   # Format for better display
    }
