#!/bin/bash
BPE_TOKENS=16000

while [ $# -gt 0 ] ; do
  case $1 in
    -s | --source) SRC="$2" ;;
    -t | --target) TGT="$2" ;;
    -a | --arch) ARCH="$2" ;;
    --vocab_size) VS="$2" ;;
  esac
  shift
done

if [ ! -z "$VS" ]; then
    BPE_TOKENS=$VS
fi

# while getopts s:t:a: flag
# do
#     case "${flag}" in
#         s) SRC=${OPTARG};;
#         t) TGT=${OPTARG};;
#         a) ARCH=${OPTARG};;
#     esac
# done

if [ -z "$SRC" ] || [ -z "$TGT" ] || [ -z "$ARCH" ] || [ -z "$VS" ]; then
    echo 'Missing -s or -t or -a (--arch) or --vocab_size' >&2
    exit 1
fi

ROOT=$(dirname "$0")
CONFIGS=$ROOT/../configs
DATA=$ROOT/../data
PROCESSED=$DATA/processed
SPM_FOLDER=$DATA/spm_models
DATA_FOLDER=$SRC-$TGT
DATA_FOLDER_BPE=${DATA_FOLDER}/${BPE_TOKENS}
SPM_FOLDER_BPE=$SPM_FOLDER/$DATA_FOLDER_BPE
mkdir -p $CONFIGS

TAG="base"
name=$SRC-$TGT-$BPE_TOKENS-$TAG_$ARCH

cat > $CONFIGS/test.yaml <<- EOC
name: "$name"

data:
    src: "$SRC"
    trg: "$TGT"
    train: "$PROCESSED/$SRC-$TGT/$BPE_TOKENS/train.bpe"
    dev:   "$PROCESSED/$SRC-$TGT/$BPE_TOKENS/dev.bpe"
    test:  "$PROCESSED/$SRC-$TGT/$BPE_TOKENS/test.bpe"
    level: "bpe"
    lowercase: True
    max_sent_length: 100
    src_vocab: "$SPM_FOLDER_BPE/vocab.txt"
    trg_vocab: "$SPM_FOLDER_BPE/vocab.txt"

testing:
    beam_size: 5
    alpha: 1.0

training:
    #load_model: "models/$name/1.ckpt" # if uncommented, load a pre-trained model from this checkpoint
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
    epochs: 1                  # TODO: Decrease for when playing around and checking of working. Around 30 is sufficient to check if its working at all
    validation_freq: 1000          # TODO: Set to at least once per epoch.
    logging_freq: 100
    eval_metric: "bleu"
    model_dir: "$ROOT/../checkpoints_joeynmt/$name"
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
        type: "$ARCH"
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
        type: "$ARCH"
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

python3 -m joeynmt train $CONFIGS/test.yaml