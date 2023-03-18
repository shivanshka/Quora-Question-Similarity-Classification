from flask import Flask, request


app= Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return "This app is under construction"

if __name__=="__main__":
    app.run()