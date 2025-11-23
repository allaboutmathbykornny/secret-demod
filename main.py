# main.py

import os
import io
import numpy as np
from flask import Flask, render_template, request, send_file
from scipy.io import wavfile

app = Flask(__name__)


def generate_sample_permutation(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    return perm


def invert_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def permute_samples(data: np.ndarray, perm: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.shape[0] != len(perm):
        raise ValueError(
            f"Length mismatch: data has {data.shape[0]} samples, "
            f"but permutation length is {len(perm)}"
        )

    if data.ndim == 1:
        result = data[perm]
    else:
        result = data[perm, :]

    return result.astype(data.dtype, copy=False)


def scramble_array(data: np.ndarray, seed: int) -> np.ndarray:
    data = np.asarray(data)
    n_samples = data.shape[0]
    perm = generate_sample_permutation(n_samples, seed)
    scrambled = permute_samples(data, perm)
    return scrambled


def unscramble_array(data: np.ndarray, seed: int) -> np.ndarray:
    data = np.asarray(data)
    n_samples = data.shape[0]
    perm = generate_sample_permutation(n_samples, seed)
    inv_perm = invert_permutation(perm)
    unscrambled = permute_samples(data, inv_perm)
    return unscrambled


def process_file_full_perm(in_bytes: bytes, seed: int):
    in_buf = io.BytesIO(in_bytes)
    fs, data = wavfile.read(in_buf)
    processed = unscramble_array(data, seed)

    out_buf = io.BytesIO()
    wavfile.write(out_buf, fs, processed)
    out_buf.seek(0)

    return out_buf, fs


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("audiofile")
        seed_text = request.form.get("seed", "").strip()

        if not f or f.filename == "":
            return "No file uploaded.", 400

        if not seed_text:
            return "No seed provided.", 400

        try:
            seed = int(seed_text)
        except ValueError:
            return "Seed must be an integer.", 400

        in_bytes = f.read()

        try:
            out_buf, fs = process_file_full_perm(in_bytes, seed)
        except Exception as e:
            return f"Error during processing: {e}", 500

        base_name, _ = os.path.splitext(f.filename)
        out_name = base_name + "_unscrambled.wav"

        return send_file(
            out_buf,
            as_attachment=True,
            download_name=out_name,
            mimetype="audio/wav"
        )

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
