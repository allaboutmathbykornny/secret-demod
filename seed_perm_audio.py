#!/usr/bin/env python3
"""
Seed-based full-sample permutation scrambler / unscrambler for WAV files.

This permutes EVERY sample (not blocks).

Usage examples:

  Scramble:
    python seed_perm_audio_full.py scramble --seed 131005 input.wav scrambled.wav

  Unscramble:
    python seed_perm_audio_full.py unscramble --seed 131005 scrambled.wav recovered.wav
"""

import argparse
import numpy as np
from scipy.io import wavfile


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

    Example:
        perm = [2, 0, 1]
        inv_perm = [1, 2, 0]
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

    # Preserve dtype
    return result.astype(data.dtype, copy=False)


def scramble_audio(data: np.ndarray, seed: int) -> np.ndarray:
    """
    Scramble audio by permuting ALL samples using a seed.
    """
    data = np.asarray(data)
    n_samples = data.shape[0]
    perm = generate_sample_permutation(n_samples, seed)
    scrambled = permute_samples(data, perm)
    return scrambled


def unscramble_audio(data: np.ndarray, seed: int) -> np.ndarray:
    """
    Unscramble audio that was scrambled with scramble_audio using the same seed.
    """
    data = np.asarray(data)
    n_samples = data.shape[0]
    perm = generate_sample_permutation(n_samples, seed)
    inv_perm = invert_permutation(perm)
    unscrambled = permute_samples(data, inv_perm)
    return unscrambled


# ---------- CLI handling ----------

def main():
    parser = argparse.ArgumentParser(
        description="Seed-based full-sample permutation scrambler / unscrambler for WAV files."
    )
    parser.add_argument(
        "mode",
        choices=["scramble", "unscramble"],
        help="Choose whether to scramble or unscramble.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Integer seed controlling the permutation (must match for unscrambling).",
    )
    parser.add_argument("input", help="Input WAV file path.")
    parser.add_argument("output", help="Output WAV file path.")

    args = parser.parse_args()

    # Read input WAV
    fs, data = wavfile.read(args.input)

    if args.mode == "scramble":
        print(f"Scrambling '{args.input}' -> '{args.output}' with seed={args.seed}")
        out_data = scramble_audio(data, seed=args.seed)
    else:  # unscramble
        print(f"Unscrambling '{args.input}' -> '{args.output}' with seed={args.seed}")
        out_data = unscramble_audio(data, seed=args.seed)

    wavfile.write(args.output, fs, out_data)
    print("Done.")


if __name__ == "__main__":
    main()
