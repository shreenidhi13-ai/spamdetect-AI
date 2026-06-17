from flask import Flask, render_template, request
import joblib
import re
import datetime
import csv
import os


app = Flask(__name__)


model = joblib.load(
    "model/spam_model.pkl"
)




spam_words = [

# Offers & Money
"free",
"offer",
"discount",
"bonus",
"cash",
"money",
"prize",
"reward",
"winner",
"winning",
"lottery",
"jackpot",
"claim",
"gift",
"coupon",
"deal",
"save",
"earn",
"income",
"profit",

# Promotions
"congratulations",
"selected",
"exclusive",
"guaranteed",
"risk free",
"no cost",
"100%",

# Account Phishing
"verify",
"verification",
"confirm",
"update",
"login",
"password",
"account",
"suspended",
"blocked",
"locked",
"security",
"authentication",
"identity",
"credentials",

# Links / Actions
"click",
"click here",
"open",
"download",
"visit",
"link",
"access",
"subscribe",
"register",
"activate",

# Urgency
"urgent",
"immediately",
"act now",
"hurry",
"limited time",
"last chance",
"final notice",
"attention",
"alert",
"warning",
"important",
"expires",
"deadline"

]



def analyze_email(email):

    text = email.lower()

    reasons = []

    score = 0


    for word in spam_words:

        if word in text:

            score += 1

            reasons.append(
                f"Suspicious word detected: {word}"
            )


    if "http" in text or "www" in text:

        score += 1

        reasons.append(
            "Contains suspicious link"
        )


    if email.isupper():

        score += 1

        reasons.append(
            "Too many capital letters"
        )


    if email.count("!") > 3:

        score += 1

        reasons.append(
            "Excessive urgency symbols"
        )


    return score, reasons




@app.route("/")
def home():

    return render_template(
        "index.html"
    )



@app.route("/predict", methods=["POST"])
def predict():


    email = request.form["email"]


    prediction = model.predict(
        [email]
    )


    risk, reasons = analyze_email(email)



    if prediction[0] == 1 or risk >= 2:


        confidence = min(
            90 + risk*2,
            99
        )


        result = "🚨 Spam Email Detected"


        status="spam"


    else:


        confidence = 95 - risk*5

        result="✅ Safe Email"

        status="safe"



    # save history

    file="history.csv"


    exists=os.path.exists(file)


    with open(
        file,
        "a",
        newline=""
    ) as f:


        writer=csv.writer(f)


        if not exists:

            writer.writerow(
            [
            "Date",
            "Email",
            "Result",
            "Confidence"
            ]
            )


        writer.writerow(
        [
        datetime.datetime.now(),
        email,
        status,
        confidence
        ]
        )




    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        reasons=reasons
    )



app.run(debug=True)