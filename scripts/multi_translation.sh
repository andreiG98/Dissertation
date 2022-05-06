# First install sacrebleu, sentencepiece, opustools, tensorboardX
pip install sacrebleu sentencepiece opustools tensorboardX

# # Then download and preprocess the data
# bash prepare-multilingual.sh

SRCS=(
    # "bg"
    "cs"
    # "da"
    # "de"
    # "el"
    "es"
    # "et"
    # "fi"
    "fr"
    "hu"
    # "it"
    # "lt"
    # "lv"
    # "nl"
    "pl"
    # "pt"
    "sk"
    "sl"
    "sv"
)
TGT=ro

ALL_SRCS=${SRCS[*]}
JOINED_SRCS=${ALL_SRCS// /_}
LANG_PAIRS="${SRCS[@]/%/-$TGT}"
LANG_PAIRS=${LANG_PAIRS// /,}
echo $LANG_PAIRS

echo $PWD

ROOT=$(dirname "$0")
# Binarize the $SRC-$TGT dataset
# TEXT=examples/translation/multilingual/processed
TEXT=$ROOT/../data/processed
echo "binarize data with fairseq-preprocess"
SRC=${SRCS[0]}
fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref $TEXT/$SRC-$TGT/train \
    --validpref $TEXT/$SRC-$TGT/dev \
    --destdir $ROOT/../data/data-bin/multilingual.$JOINED_SRCS.$TGT \
    --workers 10

for ((i=1;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    echo $SRC
    # Binarize the $SRC-$TGT dataset
    # NOTE: it's important to reuse the $TGT dictionary from the previous step
    fairseq-preprocess --source-lang $SRC --target-lang $TGT \
        --trainpref $TEXT/$SRC-$TGT/train \
        --validpref $TEXT/$SRC-$TGT/dev \
        --tgtdict $ROOT/../data/data-bin/multilingual.$JOINED_SRCS.$TGT/dict.$TGT.txt \
        --destdir $ROOT/../data/data-bin/multilingual.$JOINED_SRCS.$TGT \
        --workers 10
done

# Train a multilingual transformer model
# NOTE: the command below assumes 1 GPU, but accumulates gradients from
#       8 fwd/bwd passes to simulate training on 8 GPUs
echo "train model"
mkdir -p $ROOT/../checkpoints/multilingual_transformer_$JOINED_SRCS
CUDA_VISIBLE_DEVICES=0 fairseq-train $ROOT/../data/data-bin/multilingual.$JOINED_SRCS.$TGT/ \
    --max-epoch 50 \
    --patience 10 \
    --task multilingual_translation --lang-pairs $LANG_PAIRS \
    --arch multilingual_transformer \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir $ROOT/../checkpoints/multilingual_transformer_$JOINED_SRCS \
    --tensorboard-logdir $ROOT/../checkpoints/multilingual_transformer_$JOINED_SRCS/log-tb \
    --wandb-project dissertation \
    --max-tokens 4000 \
    --update-freq 8 \
    --fp16 \
    --memory-efficient-fp16
    # --eval-bleu \
    # --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    # --eval-bleu-detok moses \
    # --eval-bleu-remove-bpe=sentencepiece \
    # --eval-bleu-print-samples
#     # # --ddp-backend=legacy_ddp \

# # # Generate and score the test set with sacrebleu
# # SRC=de
# # # sacrebleu --test-set iwslt17 --language-pair ${SRC}-$TGT --echo src \
# # #     | python scripts/spm_encode.py --model examples/translation/multilingual.$JOINED_SRCS.$TGT/sentencepiece.bpe.model \
# # #     > iwslt17.test.${SRC}-$TGT.${SRC}.bpe
# # cat $TEXT/test.$SRC-$TGT.$SRC \
# #     | fairseq-interactive $ROOT/../data/data-bin/multilingual.$JOINED_SRCS.$TGT/ \
# #       --task multilingual_translation --lang-pairs $LANG_PAIRS \
# #       --source-lang ${SRC} --target-lang $TGT \
# #       --path checkpoints/multilingual_transformer_$JOINED_SRCS/checkpoint_best.pt \
# #       --buffer-size 2000 --batch-size 128 \
# #       --beam 5 --remove-bpe=sentencepiece \
# #     > $TEXT/test.$SRC-$TGT.$TGT.sys
# # grep ^H $TEXT/test.$SRC-$TGT.$TGT.sys | cut -f3 \
# #     | sacrebleu $TEXT/test.$SRC-$TGT.$TGT
# # # grep ^H $TEXT/test.$SRC-$TGT.$TGT.sys | cut -f3 > $TEXT/test.$SRC-$TGT.$TGT.hyp

# # # fairseq-generate $ROOT/../data/data-bin/multilingual.$JOINED_SRCS.$TGT \
# # #     --path checkpoints/multilingual_transformer_$JOINED_SRCS/checkpoint_best.pt \
# # #     --batch-size 128 --beam 5 --remove-bpe
