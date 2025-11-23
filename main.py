"""
Secret Full-Sample Permutation Scrambler – Flask web app

What it does:
- Shows a web page where you can upload a WAV file.
- Asks for a seed (integer) and a mode: "scramble" or "unscramble".
- Permutes ALL samples using a pseudo-random permutation from the seed.
- Unscramble regenerates the same permutation and applies the inverse.

Use the SAME seed to unscramble what you scrambled.
"""

import os
import io
import numpy as np
from flask import Flask, render_template, request, send_file
from scipy.io import wavfile

app = Flask(__name__)


# ---------- Permutation helpers ----------

def generate_sample_permutation(n_samples: int, seed: int) -> np.ndarray:
    """
    Generate a deterministic permutation of sample indices [0, 1, ..., n_samples-1]
    using the given seed.

    Same (n_samples, seed) -> same permutation every time.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    return perm


def invert_permutation(perm: np.ndarray) -> np.ndarray:
    """
    Given a permutation array 'perm', return its inverse.
    """
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


# ---------- Core sample permutation logic ----------

def permute_samples(data: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Apply a sample permutation to an audio array.

    data:
      - Mono: shape (N,)
      - Multi-channel: shape (N, C)

    perm:
      - Permutation array of length N.

    Returns:
      - data with its samples permuted, same shape and dtype as input.
    """
    data = np.asarray(data)
    if data.shape[0] != len(perm):
        raise ValueError(
            f"Length mismatch: data has {data.shape[0]} samples, "
            f"but permutation length is {len(perm)}"
        )

    if data.ndim == 1:
        # Mono
        result = data[perm]
    else:
        # Multi-channel: permute along time dimension
        result = data[perm, :]

    return result.astype(data.dtype, copy=False)


def scramble_array(data: np.ndarray, seed: int) -> np.ndarray:
    """
    Scramble audio by permuting ALL samples using a seed.
    """
    data = np.asarray(data)
    n_samples = data.shape[0]
    perm = generate_sample_permutation(n_samples, seed)
    scrambled = permute_samples(data, perm)
    return scrambled


def unscramble_array(data: np.ndarray, seed: int) -> np.ndarray:
    """
    Unscramble audio that was scrambled with scramble_array using the same seed.
    """
    data = np.asarray(data)
    n_samples = data.shape[0]
    perm = generate_sample_permutation(n_samples, seed)
    inv_perm = invert_permutation(perm)
    unscrambled = permute_samples(data, inv_perm)
    return unscrambled


def process_file_full_perm(in_bytes: bytes, seed: int, mode: str):
    """
    Take WAV bytes, apply full-sample permutation scrambling/unscrambling,
    and return new WAV bytes.
    """
    in_buf = io.BytesIO(in_bytes)
    fs, data = wavfile.read(in_buf)

    if mode == "scramble":
        processed = scramble_array(data, seed)
    elif mode == "unscramble":
        processed = unscramble_array(data, seed)
    else:
        raise ValueError("mode must be 'scramble' or 'unscramble'")

    out_buf = io.BytesIO()
    wavfile.write(out_buf, fs, processed)
    out_buf.seek(0)

    return out_buf, fs


# ---------- Flask routes ----------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("audiofile")
        seed_text = request.form.get("seed", "").strip()
        mode = request.form.get("mode", "scramble").lower()

        if not f or f.filename == "":
            return "No file uploaded.", 400

        if not seed_text:
            return "No seed provided.", 400

        try:
            seed = int(seed_text)
        except ValueError:
            return "Seed must be an integer.", 400

        if mode not in ("scramble", "unscramble"):
            return "Invalid mode. Choose scramble or unscramble.", 400

        in_bytes = f.read()

        try:
            out_buf, fs = process_file_full_perm(in_bytes, seed, mode)
        except Exception as e:
            return f"Error during processing: {e}", 500

        base_name, _ = os.path.splitext(f.filename)
        suffix = "_scrambled" if mode == "scramble" else "_unscrambled"
        out_name = base_name + suffix + ".wav"

        return send_file(
            out_buf,
            as_attachment=True,
            download_name=out_name,
            mimetype="audio/wav"
        )

    return render_template("index.html")


if __name__ == "__main__":
    # For local testing; on Render you'll use gunicorn via Procfile
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
