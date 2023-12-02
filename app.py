from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import os

# Initialize Flask application
app = Flask(__name__)

# Paths for model and auxiliary files (update as necessary)
root = Path(os.getcwd())
aux_re = root / 'image-captioning' / 'Inception' / 'RetrainedInceptionLSTM'
model_re_path = root / 'image-captioning' / 'Inception' / 'RetrainedInceptionLSTM' / 'Model'
model_inception_path = root / 'image-captioning' / 'Inception' / 'RetrainedInceptionFeatureExtraction' / 'Model'

# Load models and auxiliary files
word2Index = np.load(aux_re / "word2Index.npy", allow_pickle=True).item()
index2Word = np.load(aux_re / "index2Word.npy", allow_pickle=True).item()
variable_params = np.load(aux_re / "variable_params.npy", allow_pickle=True).item()
max_len = variable_params['max_caption_len']
print(model_re_path)
model_re = tf.keras.models.load_model(model_re_path)
model_inception = tf.keras.models.load_model(model_inception_path)

# Function to preprocess image for InceptionV3
def preprocess_image_inception(image):
    width, height = 299, 299
    image = image.resize(size=(width, height))
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    x = np.array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x.reshape(1, 299, 299, 3)
    return x

# Function to extract features from an image
def extract_features(model, image):
    features = model.predict(image, verbose=0)
    return features

# Beam search function
def beam_search(model, features, max_len, word2Index, index2Word, beam_index):
    start = [word2Index["startseq"]]
    start_word = [[start, 1]]

    final_preds = []
    live_seqs = beam_index
    features = np.tile(features, (beam_index,1))
    count = 0
    while len(start_word) > 0:
        #print(count)
        count+=1
        temp = []
        padded_seqs = []
        #Get padded seqs for each of the starting seqs so far, misnamed as start_word
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_len, padding='post')
            padded_seqs.append(par_caps)

        #Formatting input so that it can be used for a prediction
        padded_seqs = np.array(padded_seqs).reshape(len(start_word), max_len)

        preds = model.predict([features[:len(start_word)],padded_seqs], verbose=0)

        #Getting the best branches for each of the start seqs that we had
        for index, pred in enumerate(preds):
            word_preds = np.argsort(pred)[-live_seqs:]
            for w in word_preds:
                next_cap, prob = start_word[index][0][:], start_word[index][1]
                next_cap.append(w)
                prob *= pred[w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words from all branches
        start_word = start_word[-live_seqs:]

        for pair in start_word:
            if index2Word[pair[0][-1]] == 'endseq':
                final_preds.append([pair[0][:-1], pair[1]])
                start_word = start_word[:-1]
                live_seqs -= 1
            if len(pair[0]) == max_len:
                final_preds.append(pair)
                start_word = start_word[:-1]
                live_seqs -= 1

    # Between all the finished sequences (either max len or predicted endseq), decide which is best
    max_prob = 0
    for index, pred in enumerate(final_preds):
        if pred[1] > max_prob:
            best_index = index
            max_prob = pred[1]

    # Convert to readable text
    final_pred = final_preds[best_index]
    final_caption = [index2Word[i] for i in final_pred[0]]
    final_caption = ' '.join(final_caption[1:])
    return final_caption

# Generate caption for an image
def generate_caption(model, features, max_len, word2Index, index2Word, beam_index=3):
    caption = beam_search(model, features, max_len, word2Index, index2Word, beam_index)
    return caption

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            image = Image.open(BytesIO(file.read()))
            pp_image = preprocess_image_inception(image)
            features = extract_features(model_inception, pp_image)
            caption = generate_caption(model_re, features, max_len, word2Index, index2Word)
            return render_template('upload.html', caption=caption)
    return render_template('upload.html')

# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image = Image.open(BytesIO(file.read()))
        pp_image = preprocess_image_inception(image)
        features = extract_features(model_inception, pp_image)
        caption = generate_caption(model_re, features, max_len, word2Index, index2Word)
        caption=str(caption).strip()
        return jsonify(caption)
        #return jsonify({'caption': caption})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5000)