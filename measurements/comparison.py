# comparison_graphs.py

import matplotlib.pyplot as plt
import numpy as np

def plot_token_efficiency_comparison(delta_values, tokens_watermarking, tokens_embedding_1bit):
    """
    Plots token efficiency comparison between watermarking and 1-bit LSH embedding.

    Parameters:
    - delta_values (list or np.array): Perturbation strength values (δ) for the watermarking system.
    - tokens_watermarking (list or np.array): Number of tokens per bit for watermarking.
    - tokens_embedding_1bit (list or np.array): Number of tokens per bit for real 1-bit LSH embedding.

    Goal:
    - Compare how much text overhead each system requires per hidden bit.
    
    # TODO: Implement function to generate scatter plot.
    """
    pass


def plot_tampering_effect_comparison(tampering_levels, recovery_watermarking, recovery_embedding_1bit):
    """
    Plots message recovery accuracy for both systems under adversarial tampering.

    Parameters:
    - tampering_levels (list or np.array): Different levels of tampering (e.g., edits, synonym swaps).
    - recovery_watermarking (list or np.array): Percentage of successful recovery in watermarking system.
    - recovery_embedding_1bit (list or np.array): Percentage of successful recovery in 1-bit LSH embedding system.

    Goal:
    - Show which system is more robust under different types of tampering.
    
    # TODO: Implement function to generate line plot.
    """
    pass