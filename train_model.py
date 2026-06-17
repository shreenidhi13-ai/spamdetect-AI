import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


data = pd.read_csv(
    "../data/spam.csv",
    encoding="latin-1"
)


data = data[['v1','v2']]

data.columns = [
    'label',
    'message'
]


data['label'] = data['label'].map(
{
'ham':0,
'spam':1
}
)



def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z ]',
        '',
        text
    )

    return text



data['message'] = data['message'].apply(clean_text)



X = data['message']
y = data['label']



X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=1
)



model = Pipeline([

(
'tfidf',
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,3)
)
),

(
'classifier',
LinearSVC()
)

])



model.fit(
X_train,
y_train
)



prediction = model.predict(
X_test
)



print(
"Accuracy:",
accuracy_score(
y_test,
prediction
)
)



joblib.dump(
model,
"../model/spam_model.pkl"
)


print(
"Advanced model saved"
)