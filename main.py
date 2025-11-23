"""
main.py

Secret Demod – minimal Replit web app

What it does:
- Shows a web page where you can upload a WAV file.
- Ask for a carrier frequency (in Hz).
- Demodulates the file and sends back a new WAV you can download.

Notes:
- This version uses scipy.io.wavfile instead of soundfile
  to keep dependencies simple on Replit.
- Input must be a WAV file.
"""

import os
import io
import numpy as np
from flask import Flask, render_template, request, send_file
from scipy import signal
from scipy.io import wavfile

app = Flask(__name__)

# ---------- Demodulation logic ----------

def secret_demod_array(data, fs, carrier_freq_hz):
    """
    Demodulate a 1D numpy array audio signal at the given carrier frequency.
    Returns a float32 array in range [-1, 1].
    """
    # Convert to float if data is int16 or similar
    if np.issubdtype(data.dtype, np.integer):
        max_int = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_int
    else:
        data = data.astype(np.float32)

    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    nyquist = fs / 2.0
    n = len(data)
    t = np.arange(n) / fs

    # Mix down with cosine
    carrier = np.cos(2.0 * np.pi * carrier_freq_hz * t)
    mixed = 2.0 * data * carrier

    # Low-pass filter
    baseband_max = 15000.0
    cutoff = min(baseband_max, nyquist * 0.9)
    norm_cutoff = cutoff / nyquist

    b, a = signal.butter(6, norm_cutoff, btype="low")
    demod = signal.filtfilt(b, a, mixed)

    # Normalize to [-1, 1]
    max_abs = np.max(np.abs(demod))
    if max_abs > 0:
        demod = 0.99 * demod / max_abs

    return demod.astype(np.float32)


def secret_demod_file(in_bytes, carrier_freq_hz):
    """
    Take WAV bytes, demodulate, return new WAV bytes.
    """
    # Read from bytes
    in_buf = io.BytesIO(in_bytes)
    fs, data = wavfile.read(in_buf)

    # Demodulate
    demod = secret_demod_array(data, fs, carrier_freq_hz)

    # Convert back to int16 for WAV
    out_int16 = (demod * 32767).astype(np.int16)

    # Write to bytes
    out_buf = io.BytesIO()
    wavfile.write(out_buf, fs, out_int16)
    out_buf.seek(0)
    return out_buf, fs


# ---------- Flask routes ----------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("audiofile")
        freq_text = request.form.get("freq", "").strip()

        if not f or f.filename == "":
            return "No file uploaded.", 400

        if not freq_text:
            return "No frequency provided.", 400

        try:
            carrier_freq = float(freq_text)
        except ValueError:
            return "Frequency must be a number (in Hz).", 400

        # Read uploaded file into memory
        in_bytes = f.read()

        try:
            out_buf, fs = secret_demod_file(in_bytes, carrier_freq)
        except Exception as e:
            return f"Error during demodulation: {e}", 500

        # Name output file based on input name
        base_name, _ = os.path.splitext(f.filename)
        out_name = base_name + "_demod.wav"

        return send_file(
            out_buf,
            as_attachment=True,
            download_name=out_name,
            mimetype="audio/wav"
        )

    return render_template("index.html")


if __name__ == "__main__":
    # Replit usually sets PORT in the environment
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
