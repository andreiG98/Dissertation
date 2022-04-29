# First install sacrebleu and sentencepiece
pip install sacrebleu sentencepiece

# Then download and preprocess the data
cd examples/translation/
bash prepare-multilingual.sh
cd ../..

SRCS=(
    "de"
    "fr"
)
SRC=de
TGT=en

ALL_SRCS=${SRCS[*]}
JOINED_SRCS=${ALL_SRCS// /_}
LANG_PAIRS="${SRCS[@]/%/-$TGT}"
LANG_PAIRS=${LANG_PAIRS// /,}
echo $LANG_PAIRS

# Binarize the de-en dataset
TEXT=examples/translation/multilingual.$JOINED_SRCS.$TGT.bpe16k
# fairseq-preprocess --source-lang $SRC --target-lang $TGT \
#     --trainpref $TEXT/train.$SRC-$TGT \
#     --validpref $TEXT/valid.$SRC-$TGT \
#     --destdir data-bin/multilingual.$JOINED_SRCS.$TGT.bpe16k \
#     --workers 10

SRC=fr
# # Binarize the fr-en dataset
# # NOTE: it's important to reuse the en dictionary from the previous step
# fairseq-preprocess --source-lang $SRC --target-lang $TGT \
#     --trainpref $TEXT/train.$SRC-$TGT \
#     --validpref $TEXT/valid.$SRC-$TGT \
#     --tgtdict data-bin/multilingual.$JOINED_SRCS.$TGT.bpe16k/dict.$TGT.txt \
#     --destdir data-bin/multilingual.$JOINED_SRCS.$TGT.bpe16k \
#     --workers 10

# # Train a multilingual transformer model
# # NOTE: the command below assumes 1 GPU, but accumulates gradients from
# #       8 fwd/bwd passes to simulate training on 8 GPUs
# mkdir -p checkpoints/multilingual_transformer_$JOINED_SRCS
# CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/multilingual.$JOINED_SRCS.$TGT.bpe16k/ \
#     --max-epoch 1 \
#     --task multilingual_translation --lang-pairs $LANG_PAIRS \
#     --arch multilingual_transformer_iwslt_de_en \
#     --share-decoders --share-decoder-input-output-embed \
#     --optimizer adam --adam-betas '(0.9, 0.98)' \
#     --lr 0.0005 --lr-scheduler inverse_sqrt \
#     --warmup-updates 4000 --warmup-init-lr '1e-07' \
#     --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --save-dir checkpoints/multilingual_transformer_$JOINED_SRCS \
#     --tensorboard-logdir checkpoints/multilingual_transformer_$JOINED_SRCS/log-tb \
#     --max-tokens 4000 \
#     --update-freq 8 \
#     --fp16 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe=sentencepiece \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
#     # --ddp-backend=legacy_ddp \

# # Generate and score the test set with sacrebleu
# SRC=de
# # sacrebleu --test-set iwslt17 --language-pair ${SRC}-$TGT --echo src \
# #     | python scripts/spm_encode.py --model examples/translation/multilingual.$JOINED_SRCS.$TGT.bpe16k/sentencepiece.bpe.model \
# #     > iwslt17.test.${SRC}-$TGT.${SRC}.bpe
# cat $TEXT/test.$SRC-$TGT.$SRC \
#     | fairseq-interactive data-bin/multilingual.$JOINED_SRCS.$TGT.bpe16k/ \
#       --task multilingual_translation --lang-pairs $LANG_PAIRS \
#       --source-lang ${SRC} --target-lang $TGT \
#       --path checkpoints/multilingual_transformer_$JOINED_SRCS/checkpoint_best.pt \
#       --buffer-size 2000 --batch-size 128 \
#       --beam 5 --remove-bpe=sentencepiece \
#     > $TEXT/test.$SRC-$TGT.$TGT.sys
# grep ^H $TEXT/test.$SRC-$TGT.$TGT.sys | cut -f3 \
#     | sacrebleu $TEXT/test.$SRC-$TGT.$TGT
# # grep ^H $TEXT/test.$SRC-$TGT.$TGT.sys | cut -f3 > $TEXT/test.$SRC-$TGT.$TGT.hyp

# # fairseq-generate data-bin/multilingual.$JOINED_SRCS.$TGT.bpe16k \
# #     --path checkpoints/multilingual_transformer_$JOINED_SRCS/checkpoint_best.pt \
# #     --batch-size 128 --beam 5 --remove-bpe
