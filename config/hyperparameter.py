IS_TEST = False

if IS_TEST:
    HMM_PARAM_GRID = {
        "covariance_type": ["diag"],
        "n_components":    [5],
        "n_iter":          [1],
    }

    HSMM_PARAM_GRID = {
        "covariance_type": ["diag"],
        "n_components":    [2],
        "n_iter":          [100],
        "max_duration":    [100],
    }

    CNN_PARAM_GRID = {
        "conv_channels":  [[32, 64, 64]],
        "kernel_sizes":   [[7, 5, 3]],
        "dropout_fc":     [0.2],
        "learning_rate":  [1e-3],
        "weight_decay":   [5e-4],
        "batch_size":     [64],
        "epochs":         [10],
    }

    RNN_PARAM_GRID = {
        "rnn_type":      ["gru"],
        "hidden_size":   [32],
        "num_layers":    [1],
        "bidirectional": [False],
        "dropout_rnn":   [0.2],
        "dropout_fc":    [0.3],
        "lr":            [1e-3],
        "epochs":        [10],
        "batch_size":    [64],
        "pooling":       ["last"],
    }


    TRANSFORMER_PARAM_GRID = {
        "d_model":         [32],
        "nhead":           [2],
        "num_layers":      [1],
        "dim_feedforward": [64],
        "dropout":         [0.1],
        "dropout_fc":      [0.2],
        "lr":              [1e-3],
        "weight_decay":    [1e-4],
        "batch_size":      [64],
        "epochs":          [10],
    }

else:
    # 14
    HMM_PARAM_GRID = {
        "covariance_type": ["diag", "full"],
        "n_components":    [2, 3, 4, 5, 6, 7, 8],
        "n_iter":          [100],
    }

    # 48
    CNN_PARAM_GRID = {
        # CNN trunk
        "conv_channels": [
            [32, 64, 64],
            [64, 128, 128],
            [64, 128, 256],
        ],
        "kernel_sizes": [
            [7, 5, 3],
            [5, 5, 5],
        ],
        # Regularisation & optimiser
        "dropout_fc":    [0.2, 0.4],
        "learning_rate": [1e-3, 2e-4],
        "weight_decay":  [5e-4],
        # Training loop
        "batch_size":    [32, 64],
        "epochs":        [100],
    }

    # 144
    RNN_PARAM_GRID = {
        # architecture
        "rnn_type":      ["lstm", "gru"],
        "hidden_size":   [32, 64, 128],
        "num_layers":    [1, 2],
        # regularisation
        "bidirectional": [True, False],
        "dropout_rnn":   [0.2],
        "dropout_fc":    [0.3],
        # optimisation
        "lr":            [1e-3],
        "epochs":        [100],
        "batch_size":    [32, 64],
        # pooling
        "pooling":       ["max", "mean", "last"],
    }

    # 128
    TRANSFORMER_PARAM_GRID = {
        "d_model":         [32, 64],
        "nhead":           [4],       # nhead must divide d_model
        "num_layers":      [1, 2],
        "dim_feedforward": [64, 128],
        "dropout":         [0.1, 0.3],
        "dropout_fc":      [0.2, 0.4],
        "lr":              [1e-3, 5e-4],
        "weight_decay":    [1e-4, 1e-2],
        "batch_size":      [32],
        "epochs":          [50],
    }

    HSMM_PARAM_GRID = {
        "covariance_type": ["diag"],
        "n_components":    [2, 3, 4, 5],
        "n_iter":          [100],
        "max_duration":    [100, 200],
    }