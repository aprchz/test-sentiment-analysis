from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import swag_from
import io
import json
from json import JSONEncoder as BaseJSONEncoder
from flasgger import Swagger, LazyString
import re, pandas as pd, sqlite3, pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the custom JSON encoder
class CustomJSONEncoder(BaseJSONEncoder):
    def default(self, obj):
        if isinstance(obj, LazyString):
            return str(obj)
        return super().default(obj)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.json_encoder = CustomJSONEncoder  # Set the custom encoder

swagger_template = dict(
    info={
        'title': 'API for Sentiment Analysis--EVA001',
        'version': '1',
        'description': 'Platinum Challenge Data Science Binar Academy by Akmal eVan Ana'
    },
    host='localhost:5000'  # Adjust this if needed
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

# Connect to SQLite database
connection = sqlite3.connect(r'data.db', check_same_thread=False)

# Text preprocessing functions
def clean_text(text):
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)  # Remove URLs
    text = re.sub(r'pic.twitter.com\.\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text.lower())  # Remove unwanted characters
    text = text.replace('user', '')  # Remove the word 'user'
    text = re.sub(' +', ' ', text)  # Remove extra spaces
    text = text.replace('\n', ' ')  # Remove newlines
    return text.strip()

def standardize_text(text):
    alay_df = pd.read_sql_query('SELECT * FROM kamus_alay', connection)
    alay_dict = dict(zip(alay_df['alay'], alay_df['fix']))  # Dictionary for text standardization
    words = text.split()
    standardized_words = [alay_dict.get(word, word) for word in words]  # Standardize text
    return ' '.join(standardized_words).strip()

def remove_stopwords(text):
    stopword_df = pd.read_sql_query('SELECT * FROM stopword', connection)
    stopwords = set(stopword_df['stop'])
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def preprocess_input(text):
    original_text = text
    text = clean_text(text)
    text = standardize_text(text)
    # Uncomment the following line if you want to remove stopwords
    # text = remove_stopwords(text)
    cleaned_text = text
    return original_text, cleaned_text

# API LSTM (Text)
@swag_from("docs/lstm_input_data.yml", methods=['POST'])
@app.route('/lstm_text', methods=['POST'])
def lstm_text():
    try:
        original_text = str(request.form["lstm_text"])
        cleaned_text = preprocess_input(original_text)[1]

        loaded_model = load_model(r'lstm.h5')

        with open('tokenizer_lstm.json') as f:
            tokenizer_lstm = tokenizer_from_json(json.load(f))

        def pred_sentiment(text):
            sequences = tokenizer_lstm.texts_to_sequences([text])
            padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen if needed
            predictions = loaded_model.predict(padded_sequences, batch_size=10)
            return predictions[0]

        def pred(predictions):
            labels = ['Negatif', 'Netral', 'Positif']
            return labels[predictions.argmax()]

        predictions = pred_sentiment(cleaned_text)
        sentiment = pred(predictions)

        json_response = {
            'status_code': 200,
            'description': "Result of Sentiment Analysis using LSTM",
            'data': {
                'Text': cleaned_text,
                'Sentiment': sentiment
            },
        }

        # Save to database
        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS lstm_TEXTINPUT (original_text TEXT, text TEXT, sentiment TEXT)')
        cursor.execute('INSERT INTO lstm_TEXTINPUT (original_text, text, sentiment) VALUES (?, ?, ?)', (original_text, cleaned_text, sentiment))
        connection.commit()

        return jsonify(json_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API NN (Text)
@swag_from("docs/nn_input_data.yml", methods=['POST'])
@app.route('/nn_text', methods=['POST'])
def nn_text():
    try:
        original_text = str(request.form.get('nn_text'))
        cleaned_text = preprocess_input(original_text)[1]

        with open('feature.pkl', 'rb') as f:
            loaded_vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model_NN = pickle.load(f)

        text_vectorized = loaded_vectorizer.transform([cleaned_text])
        sentiment = model_NN.predict(text_vectorized)[0]

        json_response = {
            'status_code': 200,
            'description': "Result of Sentiment Analysis using NN",
            'data': {
                'Text': cleaned_text,
                'Sentiment': sentiment
            },
        }

        # Save to database
        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS nn_TEXTINPUT (original_text TEXT, text TEXT, sentiment TEXT)')
        cursor.execute('INSERT INTO nn_TEXTINPUT (original_text, text, sentiment) VALUES (?, ?, ?)', (original_text, cleaned_text, sentiment))
        connection.commit()

        return jsonify(json_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API LSTM (Upload Text)
@swag_from("docs/lstm_upload_data.yml", methods=['POST'])
@app.route('/lstm_upload', methods=['POST'])
def lstm_upload():
    try:
        file = request.files["lstm_upload"]
        df_csv = pd.read_csv(file, encoding="latin-1")
        df_csv['TweetClean'] = df_csv['Tweet'].apply(preprocess_input).apply(lambda x: x[1])

        loaded_model = load_model(r'lstm.h5')

        with open('tokenizer_lstm.json') as f:
            tokenizer = tokenizer_from_json(json.load(f))

        def pred_sentiment(text):
            sequences = tokenizer.texts_to_sequences([text])
            padded_sequences = pad_sequences(sequences, maxlen=100)
            predictions = loaded_model.predict(padded_sequences, batch_size=10)
            return predictions[0]

        def pred(predictions):
            labels = ['Negatif', 'Netral', 'Positif']
            return labels[predictions.argmax()]

        # Prediksi sentimen untuk setiap teks
        df_csv['Sentimen'] = df_csv['TweetClean'].apply(lambda text: pred(pred_sentiment(text)))

        # Simpan ke dalam database
        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS UPLOAD_lstm (TweetOri TEXT, TweetClean TEXT, Sentimen TEXT)')
        for _, row in df_csv.iterrows():
            cursor.execute('INSERT INTO UPLOAD_lstm (TweetOri, TweetClean, Sentimen) VALUES (?, ?, ?)', 
                           (row['Tweet'], row['TweetClean'], row['Sentimen']))
        connection.commit()

        # Simpan hasil ke file CSV
        df_csv.to_csv('hasil_sentimen_lstm.csv', index=False)

        # Simpan hasil ke file JSON
        df_csv.to_json('hasil_sentimen_lstm.json', orient='records')

        return jsonify("SUKSES")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API NN (Upload Text)
@swag_from("docs/nn_upload_data.yml", methods=['POST'])
@app.route('/nn_upload', methods=['POST'])
def nn_upload():
    try:
        file = request.files["nn_upload"]
        df_csv = pd.read_csv(file, encoding="latin-1")
        df_csv['TweetClean'] = df_csv['Tweet'].apply(preprocess_input).apply(lambda x: x[1])

        with open('feature.pkl', 'rb') as f:
            loaded_vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model_NN = pickle.load(f)

        # Prediksi sentimen
        df_csv['Sentimen'] = df_csv['TweetClean'].apply(lambda text: model_NN.predict(loaded_vectorizer.transform([text]))[0])

        # Simpan ke dalam database
        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS Upload_nn (original_text TEXT, text TEXT, sentiment TEXT)')
        for _, row in df_csv.iterrows():
            cursor.execute('INSERT INTO Upload_nn (original_text, text, sentiment) VALUES (?, ?, ?)', 
                           (row['Tweet'], row['TweetClean'], row['Sentimen']))
        connection.commit()

        # Simpan hasil ke file CSV
        df_csv.to_csv('hasil_sentimen_nn.csv', index=False)

        # Simpan hasil ke file JSON
        df_csv.to_json('hasil_sentimen_nn.json', orient='records')

        return jsonify("SUKSES")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
