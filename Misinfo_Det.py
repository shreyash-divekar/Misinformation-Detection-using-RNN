from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('newmodel.pkl', 'rb'))
# load json and create model
'''json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")'''

dataframe = pd.read_csv('cleaned_dataset.csv')
x = dataframe['statement']
y = dataframe['rating']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def misinfo_det(info):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [info]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

# Define function to render "About the System" section
@app.route('/about')
def about():
    return render_template('about.html')

# Define function to render "Accuracy" section
@app.route('/accuracy')
def accuracy():
    data = {"name": ["SVM", "LR", "RF", "LSTM", "RNN", "GCN"], "Accuracy": [82, 80, 79, 86, 82, 75]}
    return render_template('accuracy.html', data=data)

# Define function to render "Prediction" section
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        news_data = request.form['news_data']
        select_notebook_file = request.form['select_notebook_file']
        select_model = request.form['select_model']

        # Perform prediction based on selected model
        if select_notebook_file == "rnn_model":
            if select_model == "rnn":
                # Assuming predict_classes function returns prediction label
                prediction = misinfo_det(news_data)
            else:
                # Assuming predict_classes function returns prediction label
                prediction = misinfo_det(news_data)
        else:
            # Assuming make_pred function returns prediction label
            pred = misinfo_det(news_data)
            if pred[0] == 0 or pred[0] == 1:
                label_arr = ["True", "False"]
                prediction = label_arr[pred[0]]
            else:
                prediction = pred[0]

        return render_template('prediction.html', prediction=prediction)

    return render_template('prediction.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

