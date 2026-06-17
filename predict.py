import joblib
import re


model = joblib.load(
    "../model/spam_model.pkl"
)


# Important spam/phishing words
spam_words = [
    "verify",
    "password",
    "suspended",
    "click",
    "urgent",
    "winner",
    "won",
    "prize",
    "reward",
    "claim",
    "free",
    "account",
    "blocked",
    "login",
    "update"
]


while True:

    email = input("\nEnter Email: ")


    prediction = model.predict([email])


    text = email.lower()


    score = 0


    for word in spam_words:
        if word in text:
            score += 1



    if prediction[0] == 1 or score >= 2:

        confidence = min(
            90 + score*2,
            99
        )

        print(
        f"🚨 SPAM EMAIL DETECTED ({confidence}% confidence)"
        )


    else:

        print(
        "✅ SAFE EMAIL"
        )