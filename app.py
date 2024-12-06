from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load your trained model and tokenizer (update the path to your model)
model_path = r"J:\Booking Scraping\Reviews\Code\Booking Reviews Classification Model using Bert"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def convert_sentiment_label(label):
    label_names = {
        1: 'إيجابي',
        0: 'سلبي'
    }
    return label_names[label]

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment_name = ""
    if request.method == 'POST':
        sentence = request.form['sentence']

        # Tokenize and prepare the input for the model
        inputs = tokenizer(sentence, return_tensors='pt')

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted label
        predicted_label = torch.argmax(outputs.logits).item()
        sentiment_name = convert_sentiment_label(predicted_label)

    return render_template('index.html', sentiment=sentiment_name)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Disable the auto-reloader

