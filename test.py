# -*- coding: utf-8 -*-
"""
Author: Andrej Leban
Created on Sun May 29 13:05:27 2022
"""

import time

import sounddevice as sd
# import soundfile as sf
import torch

from infowavegan import WaveGANGenerator, WaveGANQNetwork

sample_rate = 32000


if __name__ == "__main__":

    # Load generator from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = WaveGANGenerator(slice_len=65536)
    G.load_state_dict(torch.load("../NNs/ciw/epoch14350_step487900_G.pt", map_location=device))
    G.to(device)

    Q = WaveGANQNetwork(num_categ=5, slice_len=65536)
    Q.load_state_dict(torch.load("../NNs/ciw/epoch14350_step487900_Q.pt", map_location=device))
    Q.to(device)

    # Generate from random noise
    for i in range(100):
        z = torch.FloatTensor(1, 100).uniform_(-1, 1).to(device)
        genData = G(z)[0, 0, :].detach().cpu().numpy()
        # write(f'out.wav', sample_rate, (genData * 32767).astype(np.int16))
        sd.play(genData, sample_rate)
        time.sleep(1)
