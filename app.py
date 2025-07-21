from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open('netflix_rf_model.pkl', 'rb'))
tfidf = pickle.load(open('netflix_tfidf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    category = request.form['category']
    description = request.form['description']

    # Combine and transform input
    combined = category + ' ' + description
    input_vector = tfidf.transform([combined])

    # Make prediction
    prediction = model.predict(input_vector)[0]
    result = 'Movie' if prediction == 0 else 'TV Show'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
#%%
