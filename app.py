from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nlpaug.augmenter.word as naw
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    sentiment = predict_sentiment(review)
    sentiment_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return jsonify({'sentiment': sentiment_label[sentiment]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
