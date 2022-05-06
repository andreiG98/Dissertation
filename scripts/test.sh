cat > test.yaml <<- EOC
name: "{source_language}-{target_language}-{tag}-{vocab_size}_transformer"

data:
    src: "{source_language}"
    trg: "{target_language}"
    train: "{data_dir}/{name}/train.bpe"
    dev:   "{data_dir}/{name}/dev.bpe"
    test:  "{data_dir}/{name}/test.bpe"
    level: "bpe"
    lowercase: False
    max_sent_length: 100
    src_vocab: "{data_dir}/{name}/vocab.txt"
    trg_vocab: "{data_dir}/{name}/vocab.txt"

testing:
    beam_size: 5
    alpha: 1.0

training:
    #load_model: "models/{name}_transformer/1.ckpt" # if uncommented, load a pre-trained model from this checkpoint
    random_seed: 42
    optimizer: "adam"
    normalization: "tokens"
    adam_betas: [0.9, 0.999] 
    scheduling: "plateau"           # TODO: try switching from plateau to Noam scheduling
    patience: 5                     # For plateau: decrease learning rate by decrease_factor if validation score has not improved for this many validation rounds.
    learning_rate_factor: 0.5       # factor for Noam scheduler (used with Transformer)
    learning_rate_warmup: 1000      # warmup steps for Noam scheduler (used with Transformer)
    decrease_factor: 0.7
    loss: "crossentropy"
    learning_rate: 0.0003
    learning_rate_min: 0.00000001
    weight_decay: 0.0
    label_smoothing: 0.1
    batch_size: 4096
    batch_type: "token"
    eval_batch_size: 3600
    eval_batch_type: "token"
    batch_multiplier: 1
    early_stopping_metric: "ppl"
    epochs: 5                  # TODO: Decrease for when playing around and checking of working. Around 30 is sufficient to check if its working at all
    validation_freq: 1000          # TODO: Set to at least once per epoch.
    logging_freq: 100
    eval_metric: "bleu"
    model_dir: "models/{name}_transformer"
    overwrite: True              # TODO: Set to True if you want to overwrite possibly existing models. 
    shuffle: True
    use_cuda: True
    max_output_length: 100
    print_valid_sents: [0, 1, 2, 3]
    keep_last_ckpts: 3

model:
    initializer: "xavier"
    bias_initializer: "zeros"
    init_gain: 1.0
    embed_initializer: "xavier"
    embed_init_gain: 1.0
    tied_embeddings: True
    tied_softmax: True
    encoder:
        type: "transformer"
        num_layers: 6
        num_heads: 4             # TODO: Increase to 8 for larger data.
        embeddings:
            embedding_dim: 256   # TODO: Increase to 512 for larger data.
            scale: True
            dropout: 0.2
        # typically ff_size = 4 x hidden_size
        hidden_size: 256         # TODO: Increase to 512 for larger data.
        ff_size: 1024            # TODO: Increase to 2048 for larger data.
        dropout: 0.3
    decoder:
        type: "transformer"
        num_layers: 6
        num_heads: 4              # TODO: Increase to 8 for larger data.
        embeddings:
            embedding_dim: 256    # TODO: Increase to 512 for larger data.
            scale: True
            dropout: 0.2
        # typically ff_size = 4 x hidden_size
        hidden_size: 256         # TODO: Increase to 512 for larger data.
        ff_size: 1024            # TODO: Increase to 2048 for larger data.
        dropout: 0.3
EOC