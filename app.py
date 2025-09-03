from flask import Flask, request, jsonify, render_template
from comic_recommender import recommend  # your function (see next section)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------- HTML page ----------
@app.route("/")
def home():
    return render_template("index.html")

# ---------- JSON API ----------
@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    data = request.get_json(force=True)
    title = data.get("title", "").strip()
    recommendations = recommend(title)
    return jsonify({"input": title, "recommendations": recommendations})

if __name__ == "__main__":
    # enable hot‑reload in dev
    app.run(debug=True, host="0.0.0.0", port=5000)
