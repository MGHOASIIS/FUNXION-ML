HMM_PARAM_GRID = {
    "covariance_type": ["diag", "full"],
    "n_components": [2, 3, 4, 5],
    "n_iter": [50, 100],
}

CNN_PARAM_GRID = {
    # ── CNN trunk ────────────────────────────────────────────
    "conv_channels": [
        [32, 64, 64],          # lighter model
        [64, 128, 128],        # baseline
        [64, 128, 256]         # deeper final block
    ],
    "kernel_sizes": [
        [7, 5, 3],             # wide → narrow
        [5, 5, 5]              # symmetric filters
    ],
    
    # ── GRU head ─────────────────────────────────────────────
    "gru_hidden":     [64, 128, 256],        # capacity sweep [128]
    "bidirectional":  [False, True],      # causal vs. acausal
    
    # ── Regularisation & optimiser ───────────────────────────
    "dropout_fc":     [0.2, 0.4],         # light vs. 0.4 strong dropout
    "learning_rate":  [1e-3, 2e-4], # start, 5e-4 half, 2e-4 one-fifth
    "weight_decay":   [5e-4],       # mild vs. moderate L2 , 5e-4
    
    # ── Training loop tweaks (optional) ─────────────────────
    "batch_size":     [32, 64],               # 16 RAM vs. 32 stability
    "warmup_epochs":  [5],                # keep fixed
    "finetune_epochs":[15]                # keep fixed (early-stop anyway)
}


RNN_PARAM_GRID = {
    # architecture
    "rnn_type":      ["lstm", "gru"],    # 2×
    "hidden_size":   [32, 64, 128],          # 2×
    "num_layers":    [1, 2],             # 2×

    # regularisation
    "bidirectional": [True, False],             # keep fixed (➜ doubles params)
    "dropout_rnn":   [0.2],              # after each RNN layer
    "dropout_fc":    [0.3],              # before final FC

    # optimisation
    "lr":            [1e-3],             # Adam learning-rate
    "epochs":        [30],               # training epochs
    "batch_size":    [32, 64],               # mini-batch size

    # sequence pooling
    "pooling":       ["max", "mean", "last"],   # 2× last
}