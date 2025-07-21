This project uses Machine Learning and Natural Language Processing (NLP) to predict whether a given Netflix title is a Movie or a TV Show, based on its category and description.
We used the Netflix Movies and TV Shows dataset from Kaggle, focusing on two key features: listed_in (representing category/genre) and description. 
The text data was preprocessed by converting to lowercase, removing stopwords, and applying TF-IDF Vectorization to transform the text into numerical features.
A Random Forest Classifier was then trained on this data to distinguish between Movies and TV Shows.
Model performance was evaluated using metrics like Accuracy, Precision, Recall, F1-Score, and visualized through a Confusion Matrix and Heatmap.
The model is deployed using Flask, offering a simple web interface where users can input a title’s metadata and get instant predictions.

