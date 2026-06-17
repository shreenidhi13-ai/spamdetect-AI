import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt



model = joblib.load(
    "model/spam_model.pkl"
)


data = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)


data = data[['v1','v2']]

data.columns = [
    "label",
    "message"
]


data['label'] = data['label'].map(
{
"ham":0,
"spam":1
}
)


X=data['message']
y=data['label']


X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



prediction=model.predict(
    X_test
)



print(
"Accuracy:",
accuracy_score(y_test,prediction)
)


print(
"Precision:",
precision_score(y_test,prediction)
)


print(
"Recall:",
recall_score(y_test,prediction)
)


print(
"F1 Score:",
f1_score(y_test,prediction)
)



cm=confusion_matrix(
    y_test,
    prediction
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)


plt.title(
"Spam Detection Confusion Matrix"
)


plt.savefig(
"confusion_matrix.png"
)


print(
"Confusion matrix saved"
)