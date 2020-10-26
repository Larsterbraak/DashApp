"""
MSc Thesis Quantitative Finance
Title: Interest rate risk due to EONIA-ESTER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: May 25th 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Create simulation for the EONIA short rate and backtest the VaR
using the Basel Committee's Traffic Light coverage test   
(1) Perform coverage test Basel
 -
(2) Perform realness classification of ESTER + 8.5 bps wrt EONIA

Inputs
(1) EONIA, calibrated TimeGAN models
-
- 

Outputs
- Classification for the Value-at-Risk model
"""

import numpy as np
from assets.training import RandomGenerator
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def load_models(epoch, hparams, hidden_dim):        
    from models.Discriminator import Discriminator
    from models.Recovery import Recovery
    from models.Generator import Generator
    from models.Embedder import Embedder
    from models.Supervisor import Supervisor
     
    if epoch % 50 != 0:
        return 'Only insert epochs that are divisible by 50.'
    else:
        # Only use when you want to load the models
        e_model_pre_trained = Embedder('logs/e_model_pre_train', hparams, hidden_dim, dimensionality = 11)
        e_model_pre_trained.load_weights('assets/ALL/embedder/epoch_' + str(epoch)).expect_partial()
        e_model_pre_trained.build([])
        
        r_model_pre_trained = Recovery('logs/r_model_pre_train', hparams, hidden_dim, dimensionality = 11)
        r_model_pre_trained.load_weights('assets/ALL/recovery/epoch_' + str(epoch)).expect_partial()
        r_model_pre_trained.build([])
        
        s_model_pre_trained = Supervisor('logs/s_model_pre_train', hparams, hidden_dim)
        s_model_pre_trained.load_weights('assets/ALL/supervisor/epoch_' + str(epoch)).expect_partial()
        s_model_pre_trained.build([])
        
        g_model_pre_trained = Generator('logs/g_model_pre_train', hparams, hidden_dim)
        g_model_pre_trained.load_weights('assets/ALL/generator/epoch_' + str(epoch)).expect_partial()
        g_model_pre_trained.build([])
        
        d_model_pre_trained = Discriminator('logs/d_model_pre_train', hparams, hidden_dim) 
        d_model_pre_trained.load_weights('assets/ALL/discriminator/epoch_' + str(epoch)).expect_partial()
        d_model_pre_trained.build([])
        
        return e_model_pre_trained, r_model_pre_trained, s_model_pre_trained, g_model_pre_trained, d_model_pre_trained