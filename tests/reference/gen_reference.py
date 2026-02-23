#!/usr/bin/env python3
"""Generate golden reference outputs from HuggingFace models for testing IMP."""

# TODO: implement reference generation
# - Load model with transformers
# - Run forward pass on fixed prompts
# - Save intermediate tensors (embeddings, attention, norms) as numpy arrays
# - Save token IDs for greedy decoding

import sys
import os


def generate_reference():
    """Generate reference data from a HuggingFace model.

    Steps to implement:
    1. Load a small model (e.g., TinyLlama) with transformers
    2. Define fixed test prompts
    3. Tokenize prompts and save token IDs
    4. Run forward pass with output_hidden_states=True, output_attentions=True
    5. Extract and save:
       - Token embeddings after embedding layer
       - Attention scores per layer
       - Hidden states after each transformer block
       - RMSNorm intermediate values
       - Final logits
    6. Run greedy decoding for N tokens and save output IDs
    7. Save all arrays as .npy files in a structured directory
    """
    print("Reference generation not yet implemented")
    return 1


if __name__ == "__main__":
    sys.exit(generate_reference())
