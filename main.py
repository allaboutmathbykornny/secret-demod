"""
Secret Permutation Scrambler – Flask web app

What it does:
- Shows a web page where you can upload a WAV file.
- Asks for a seed (integer) and a mode: "scramble" or "unscramble".
- Splits the audio into blocks and permutes those blocks using a
  pseudo-random permutation determined by the seed.

Important:
- Use the SAME seed to unscramble that you used to scramble.
- Scramble and Unscramble are perfect inverses (up to padding at the end).
"""

import os
import io
import numpy as np
from flask import Flask, render_template, request, send_file
from scipy.io import wavfile

app = Flask(__name__)

# You can tweak this: number of samples per block.
# At 44100 Hz, 44100 samples ≈ 1 second blocks, 4410 ≈ 0.1 s, etc.
BLOCK_SIZE = 44100  # 1 second at 44.1 kHz


# ---------- Core permutation logic ----------

def permute_blocks(data, seed, mode="scramble", block_size=BLOCK_SIZE):
    """
    Permute audio blocks using a deterministic permutation based on 'seed'.

    - data: numpy array of shape (N,) for mono or (N, C) for multi-channel.
    - seed: integer seed for RNG.
    - mode: "scramble" or "unscramble".
    - block_size: samples per block (same for scramble and unscramble).

    Returns a new array with the same dtype and shape (up to padding).
    """
    # Ensure data is at least 1D
    data = np.asarray(data)
    original_length = data.shape[0]

    # Pad so the length is a multiple of block_size
    n_samples = original_length
    n_blocks = int(np.ceil(n_samples / block_size))
    padded_length = n_blocks * block_size
    pad_len = padded_length - n_samples

    if data.ndim == 1:
        # Mono: shape (N,) -> pad to (padded_length,)
        if pad_len > 0:
            data_padded = np.pad(data, (0, pad_len), mode="constant", constant_values=0)
        else:
            data_padded = data
        # Reshape to (n_blocks, block_size)
        blocks = data_padded.reshape(n_blocks, block_size)
    else:
        # Multi-channel: shape (N, C)
        n_channels = data.shape[1]
        if pad_len > 0:
            data_padded = np.pad(
                data,
                ((0, pad_len), (0, 0)),
                mode="constant",
                constant_values=0
            )
        else:
            data_padded = data
        # Reshape to (n_blocks, block_size, n_channels)
        blocks = data_padded.reshape(n_blocks, block_size, n_channels)

    # Make a seed-based permutation of block indices
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_blocks)

    if mode == "scramble":
        # Scramble: reorder blocks according to perm
        permuted_blocks = blocks[perm]
    elif mode == "unscramble":
        # Unscramble: apply inverse permutation
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(n_blocks)
        permuted_blocks = blocks[inv_perm]
    else:
        raise ValueError("mode must be 'scramble' or 'unscramble'")

    # Reshape back to the padded shape
    if data.ndim == 1:
        result_padded = permuted_blocks.reshape(padded_length)
    else:
        result_padded = permuted_blocks.reshape(padded_length, -1)

    # Crop back to the original length
    result = result_padded[:original_length]

    # Preserve original dtype
    if np.issubdtype(data.dtype, np.integer):
        # Make sure we don't overflow when casting back
        info = np.iinfo(data.dtype)
        result = np.clip(result, info.min, info.max).astype(data.dtype)
    else:
        result = result.astype(data.dtype)

    return result


def process_file_with_seed_permutation(in_bytes, seed, mode):
    """
    Take WAV bytes, apply permutation-based scrambling or unscrambling,
    and return new WAV bytes.
    """
    in_buf = io.BytesIO(in_bytes)
    fs, data = wavfile.read(in_buf)

    # Apply permutation
    processed = permute_blocks(data, seed=seed, mode=mode)

    # Write to WAV in the same dtype as the processed array
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

        # Read uploaded file into memory
        in_bytes = f.read()

        try:
            out_buf, fs = process_file_with_seed_permutation(in_bytes, seed, mode)
        except Exception as e:
            return f"Error during processing: {e}", 500

        # Name output file based on input name and mode
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
    # Local testing; on Render you'll use gunicorn via Procfile
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
