import pickle

from flask import Flask, request

app = Flask(__name__)


def validate_input(sepal_length, sepal_width, petal_length, petal_width):
    if not isinstance(sepal_length, float) or (
        sepal_length < 4.3 or sepal_length > 7.9
    ):
        return False
    if not isinstance(sepal_width, float) or (sepal_width < 2.0 or sepal_width > 4.4):
        return False
    if not isinstance(petal_length, float) or (
        petal_length < 1.0 or petal_length > 6.9
    ):
        return False
    if not isinstance(petal_width, float) or (petal_width < 0.1 or petal_width > 2.5):
        return False
    return True


def predict(sepal_length, sepal_width, petal_length, petal_width):
    with open("best_model.pkl", "rb") as f:
        clf = pickle.load(f)
    return clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])


# validar inputs antes de chamar o modelo
@app.route("/predict", methods=["POST"])
def hello_world():
    args = request.get_json(force=True)
    sepal_length = args["sepal_length"]
    sepal_width = args["sepal_width"]
    petal_length = args["petal_length"]
    petal_width = args["petal_width"]
    if validate_input(sepal_length, sepal_width, petal_length, petal_width):
        result = predict(sepal_length, sepal_width, petal_length, petal_width)
        if result == 0:
            return {"class": "Iris-setosa"}
        elif result == 1:
            return {"class": "Iris-versicolor"}
    else:
        return "Invalid input"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
