import torch
import torch.nn.functional as F  # Добавляем импорт F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_WEIGHTS_PATH = 'trained_distilbert_pytorch.pt'  # <--- Path to saved model .pt
LOCAL_ASSETS_PATH = 'local_model_assets_multi'  # <--- Path to config.json etc
NUM_CLASSES = 19  # <--- Number of classes which should be predicted, model has been trained for Y

CLASS_MAPPING = {
    0: 'ABAP', 1: 'BC', 2: 'CFIN', 3: 'CMA',
    4: 'CON', 5: 'EWM', 6: 'FIN',
    7: 'MAC', 8: 'MDM', 9: 'MET',
    10: 'MNF', 11: 'O2C', 12: 'P2P',
    13: 'PMN', 14: 'RE', 15: 'REP',
    16: 'RSP', 17: 'SL', 18: 'TM'}  # <--- Dict for classes with description

if len(CLASS_MAPPING) != NUM_CLASSES:
    raise ValueError("CLASS_MAPPING and NUM_CLASSES include different number of classes")

model = None
tokenizer = None

try:
    model = DistilBertForSequenceClassification.from_pretrained(LOCAL_ASSETS_PATH, num_labels=NUM_CLASSES)
    tokenizer = DistilBertTokenizer.from_pretrained(LOCAL_ASSETS_PATH)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()
    print(f"Modle has been loaded locally with ({NUM_CLASSES} classes).")
except Exception as e:
    print(f"Error during loading local resources: {e}")
    model = None
    tokenizer = None


@app.route('/')
def index():
    return render_template('index.html', prediction_text='')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return render_template('index.html', prediction_text="Error: model has not been loaded. Check log's")

    text1 = request.form['text1']
    text2 = request.form['text2']
    combined_text = text1 + " " + text2
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probabilities = F.softmax(logits, dim=1)  # <--- Get probabilities

        max_probability, prediction_index_tensor = torch.max(probabilities, dim=1)  # <--- Get class
        prediction_index = prediction_index_tensor.item()

        confidence = max_probability.item()  # <--- Extract prob like an int

    predicted_class_name = CLASS_MAPPING.get(prediction_index, "Unknown class")

    result = f"Service line: {predicted_class_name} (Probability: {confidence * 100:.2f}%)"

    return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
