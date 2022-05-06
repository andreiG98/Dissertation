#!/bin/bash

# First install sacrebleu, sentencepiece, opustools, tensorboardX
pip install sacrebleu sacremoses sentencepiece opustools tensorboardX

while getopts s:t:a: flag
do
    case "${flag}" in
        s) SRC=${OPTARG};;
        t) TGT=${OPTARG};;
        a) ARCH=${OPTARG};;
    esac
done

if [ -z "$SRC" ] || [ -z "$TGT" ] || [ -z "$ARCH" ]; then
        echo 'Missing -s or -t or -a(architecture)' >&2
        exit 1
fi

# ARCH="transformer"
ROOT=$(dirname "$0")
TEXT=$ROOT/../data/processed

# # Then download and preprocess the data
# bash prepare-pairwise.sh -s $SRC -t $TGT

# ALL_SRCS=${SRCS[*]}
# JOINED_SRCS=${ALL_SRCS// /_}
# LANG_PAIRS="${SRCS[@]/%/-$TGT}"
# LANG_PAIRS=${LANG_PAIRS// /,}
# echo $LANG_PAIRS

# # Binarize the $SRC-$TGT dataset
# echo "binarize data with fairseq-preprocess"
# fairseq-preprocess --source-lang $SRC --target-lang $TGT \
#     --trainpref $TEXT/$SRC-$TGT/train \
#     --validpref $TEXT/$SRC-$TGT/dev \
#     --destdir $ROOT/../data/data-bin/$SRC-$TGT \
#     --workers 10

# Train the $ARCH model
# NOTE: the command below assumes 1 GPU, but accumulates gradients from
#       8 fwd/bwd passes to simulate training on 8 GPUs
echo "train model"
mkdir -p $ROOT/../checkpoints/$ARCH.$SRC-$TGT
CUDA_VISIBLE_DEVICES=0 fairseq-train $ROOT/../data/data-bin/$SRC-$TGT/ \
    --max-epoch 50 \
    --arch $ARCH \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir $ROOT/../checkpoints/$ARCH.$SRC-$TGT \
    --restore-file $ROOT/../checkpoints/$ARCH.$SRC-$TGT/checkpoint_best.pt \
    --tensorboard-logdir $ROOT/../checkpoints/$ARCH.$SRC-$TGT/log-tb \
    --wandb-project dissertation \
    --max-tokens 4000 \
    --fp16 \
    --memory-efficient-fp16 \
    --update-freq 8 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe=sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

# # # Generate and score the test set with sacrebleu
# # # SRC=de
# # # # sacrebleu --test-set iwslt17 --language-pair ${SRC}-$TGT --echo src \
# # # #     | python scripts/spm_encode.py --model examples/translation/multilingual.$JOINED_SRCS.$TGT/sentencepiece.bpe.model \
# # # #     > iwslt17.test.${SRC}-$TGT.${SRC}.bpe
# # # cat $TEXT/test.$SRC-$TGT.$SRC \
# # #     | fairseq-interactive $ROOT/../data/data-bin/$JOINED_SRCS.$TGT/ \
# # #       --task multilingual_translation --lang-pairs $LANG_PAIRS \
# # #       --source-lang ${SRC} --target-lang $TGT \
# # #       --path checkpoints/$ARCH.$SRC-$TGT/checkpoint_best.pt \
# # #       --buffer-size 2000 --batch-size 128 \
# # #       --beam 5 --remove-bpe=sentencepiece \
# # #     > $TEXT/test.$SRC-$TGT.$TGT.sys
# # # grep ^H $TEXT/test.$SRC-$TGT.$TGT.sys | cut -f3 \
# # #     | sacrebleu $TEXT/test.$SRC-$TGT.$TGT
# # # # grep ^H $TEXT/test.$SRC-$TGT.$TGT.sys | cut -f3 > $TEXT/test.$SRC-$TGT.$TGT.hyp

# # # # fairseq-generate $ROOT/../data/data-bin/$JOINED_SRCS.$TGT \
# # # #     --path checkpoints/$ARCH.$SRC-$TGT/checkpoint_best.pt \
# # # #     --batch-size 128 --beam 5 --remove-bpe
