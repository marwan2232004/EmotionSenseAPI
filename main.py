import numpy as np
from fastapi import FastAPI
import tensorflow as tf
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import io
import re
import nltk
import os
import gdown


# Function to get the model and tokenizer from google drive instead of putting it in the repo
def download_file_from_google_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)


download_file_from_google_drive('1qWkyNQXhcwlE-enuY0suIvsOy-5oRIkQ', './COVID_NLP5.keras')
download_file_from_google_drive('12_AgHa0hiIPLeWQy51yovk8hpN6xzlo5', './tokenizer5.joblib')

# Define the directory where NLTK data will be stored

nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")

# Add this directory to NLTK's data path
nltk.data.path.append(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir)  # At first you have to download these nltk packages.
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stop_words = stopwords.words('english')
stop_words.remove('not')
lemmatizer = WordNetLemmatizer()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

model = tf.keras.models.load_model('./COVID_NLP5.keras')
tokenizer = joblib.load('./tokenizer5.joblib')


def nlp_preprocessing(tweet):
    # Data cleaning
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)  # Remove hashtags
    tweet = re.sub(r'https?://\S+|www\.\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'<.*?>', '', tweet)  # Remove HTML tags
    tweet = re.sub('[^A-Za-z]+', ' ', tweet)  # Keep only alphabetic characters

    # Convert to lowercase
    tweet = tweet.lower()

    # Tokenization
    tokens = nltk.word_tokenize(tweet)  # Convert text to tokens

    # Remove single-character tokens (except meaningful ones like 'i' and 'a')
    tokens = [word for word in tokens if len(word) > 1]

    # Remove stopwords
    tweet = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tweet = [lemmatizer.lemmatize(word) for word in tweet]

    # Join words back into a single string
    tweet = ' '.join(tweet)

    # return
    return tweet


@app.get("/")
async def root():
    return {"message": "Welcome to Emotion Sense API"}


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: PredictRequest):
    data = request.text
    text = nlp_preprocessing(data)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=261, padding='post')
    prediction = np.argmax(model.predict([text]))
    return {"prediction": int(prediction)}
