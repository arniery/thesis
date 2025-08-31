# coding: utf-8
"""
Synthesis waveform from trained WaveNet.

Modified from https://github.com/r9y9/wavenet_vocoder
"""

import torch
from tqdm import tqdm
import librosa
import numpy as np
from hparams import hparams
from wavenet_vocoder import builder

# Use up to 4 threads (can adjust depending on performance)
torch.set_num_threads(4)

# Set device: prefer MPS (Apple GPU), fallback to CPU
device = torch.device("cpu")

def build_model():
    model = getattr(builder, hparams.builder)(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        weight_normalization=hparams.weight_normalization,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_scales=hparams.upsample_scales,
        freq_axis_kernel_size=hparams.freq_axis_kernel_size,
        scalar_input=True,
        legacy=hparams.legacy,
    )
    return model


def wavegen(model, c=None, tqdm=tqdm):
    """Generate waveform samples by WaveNet."""
    model.eval()
    model.make_generation_fast_()

    # Convert to tensor
    if isinstance(c, np.ndarray):
        c = torch.tensor(c, dtype=torch.float32, device=device)

    # Reshape to [1, cin_channels, T]
    if c.ndim == 2:
        c = c.permute(1, 0).unsqueeze(0)
    elif c.ndim == 3 and c.shape[0] == 1:
        c = c.permute(0, 2, 1)

    assert c.shape[0] == 1 and c.shape[1] == hparams.cin_channels, f"Unexpected shape: {c.shape}"

    Tc = c.shape[2]
    upsample_factor = hparams.hop_size
    length = Tc * upsample_factor

    initial_input = torch.zeros(1, 1, 1, device=device)
    c = c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input,
            c=c,
            g=None,
            T=length,
            tqdm=tqdm,
            softmax=True,
            quantize=True,
            log_scale_min=hparams.log_scale_min
        )

    return y_hat.view(-1).cpu().numpy()
