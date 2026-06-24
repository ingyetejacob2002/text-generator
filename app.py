from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load model once at startup
lm_generator = pipeline("text-generation", model="distilgpt2")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt.strip():
        return jsonify({"error": "Empty prompt"}), 400

    output = lm_generator(
        prompt,
        max_length=80,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_p=0.95
    )

    return jsonify({
        "result": output[0]["generated_text"]
    })


if __name__ == "__main__":
    app.run(debug=True)
