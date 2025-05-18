from flask import Flask, request, jsonify
import numpy as np
import scipy.stats as stats

app = Flask(__name__)

@app.route("/meta/summary", methods=["POST"])
def compute_meta_summary():
    data = request.get_json()
    effect_sizes = np.array(data.get("effect_sizes"))
    variances = np.array(data.get("variances"))

    if len(effect_sizes) != len(variances):
        return jsonify({"error": "Effect sizes and variances must be of the same length."}), 400

    weights = 1 / variances
    weighted_mean = np.sum(weights * effect_sizes) / np.sum(weights)
    weighted_var = 1 / np.sum(weights)
    ci_low = weighted_mean - 1.96 * np.sqrt(weighted_var)
    ci_high = weighted_mean + 1.96 * np.sqrt(weighted_var)

    return jsonify({
        "summary_effect": float(weighted_mean),
        "95CI": [float(ci_low), float(ci_high)],
        "variance": float(weighted_var)
    })

@app.route("/meta/heterogeneity", methods=["POST"])
def compute_heterogeneity():
    data = request.get_json()
    effect_sizes = np.array(data.get("effect_sizes"))
    variances = np.array(data.get("variances"))

    weights = 1 / variances
    summary_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    Q = np.sum(weights * (effect_sizes - summary_effect) ** 2)
    df = len(effect_sizes) - 1
    p_value = 1 - stats.chi2.cdf(Q, df)

    return jsonify({
        "Q": float(Q),
        "degrees_of_freedom": df,
        "p_value": float(p_value)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
