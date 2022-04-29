#!/bin/bash

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
SPM_TRAIN=$BPEROOT/learn_bpe.py
SPM_ENCODE=$BPEROOT/apply_bpe.py
BPE_TOKENS=16000

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
ARCHIVES=(
    "training-parallel-europarl-v7.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "training/europarl-v7.fr-en"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

SRCS=(
    "pt"
    "es"
)
TGT=ro
ALL_SRCS=${SRCS[*]}
JOINED_SRCS=${ALL_SRCS// /_}
# ORIG=multilingual_orig_$JOINED_SRCS
# ORIG=multilingual_orig
# DATA=multilingual.$JOINED_SRCS.$TGT
DATA=multilingual
TMP=$DATA/tmp
ORIG=$DATA/orig
TMP_DOWNLOADS=$ORIG/tmp_downloads

mkdir -p $TMP_DOWNLOADS $ORIG $TMP $DATA

cd $ORIG

# TMP_DOWNLOADS=tmp_downloads
# mkdir -p $TMP_DOWNLOADS

for SRC in "${SRCS[@]}"; do
    # download and extract data
    DATA_FOLDER=$SRC-$TGT
    echo "$DATA_FOLDER"
    if [ -d "$DATA_FOLDER" ]; then
        echo "$DATA_FOLDER already exists, skipping download"
    else
        mkdir -p $DATA_FOLDER
        opus_express -s $SRC -t $TGT -c Europarl -q \
            --download-dir tmp_downloads/ \
            --test-set $DATA_FOLDER/test \
            --dev-set $DATA_FOLDER/dev \
            --train-set $DATA_FOLDER/train
    fi
done

rm -rf $TMP_DOWNLOADS

# # download and extract data
# for ((i=0;i<${#URLS[@]};++i)); do
#     ARCHIVE=${ARCHIVES[i]}
#     if [ -f "$ARCHIVE" ]; then
#         echo "$ARCHIVE already exists, skipping download"
#     else
#         URL=${URLS[i]}
#         wget "$URL"
#         if [ -f "$ARCHIVE" ]; then
#             echo "$URL successfully downloaded."
#         else
#             echo "$URL not successfully downloaded."
#             exit 1
#         fi
#     fi
#     FILE=${ARCHIVE: -4}
#     if [ -e $FILE ]; then
#         echo "$FILE already exists, skipping extraction"
#     else
#         if [ $FILE == ".tgz" ]; then
#             tar zxvf $ARCHIVE
#         elif [ $FILE == ".tar" ]; then
#             tar xvf $ARCHIVE
#         fi
#         # tar -xzvf $ARCHIVE
#     fi
# done

cd ../..


echo "pre-processing train/dev data..."
for SRC in "${SRCS[@]}"; do
    for LANG in $SRC $TGT; do
        DATA_FOLDER=${SRC}-${TGT}

        mkdir -p $TMP/$DATA_FOLDER

        rm $TMP/$TOK

        TOK=$DATA_FOLDER/train.tags.tok.$LANG
        echo "$ORIG/$DATA_FOLDER/train.$LANG"
        # cat $ORIG/$DATA_FOLDER/train.$LANG
        for phase in "train" "dev"; do
            cat $ORIG/$DATA_FOLDER/$phase.$LANG | \
                perl $NORM_PUNC $LANG | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 8 -a -l $LANG >> $TMP/$TOK
        done
    done
done


# echo "pre-processing train data..."
# for SRC in "${SRCS[@]}"; do
#     for LANG in $SRC $TGT; do
#         EXT=${SRC}-${TGT}
#         TOK=train.tags.$EXT.tok.$LANG
#         rm $TMP/$TOK
#         for f in "${CORPORA[@]}"; do
#             cat $ORIG/$f.$LANG | \
#                 perl $NORM_PUNC $LANG | \
#                 perl $REM_NON_PRINT_CHAR | \
#                 perl $TOKENIZER -threads 8 -a -l $LANG >> $TMP/$TOK
#         done
#     done
# done


# # Clean inainte sau dupa BPE???
# ### for SRC in "${SRCS[@]}"; do
# #     EXT=${SRC}-${TGT}
# #     perl $CLEAN -ratio 1.5 $TMP/train.tags.$EXT.tok $SRC $TGT $TMP/train.tags.$EXT.tok 1 250
# #     perl $CLEAN -ratio 1.5 $TMP/valid.tags.$EXT.tok $SRC $TGT $TMP/valid.tags.$EXT.tok 1 250
# #### done

# echo "pre-processing test data..."
# for SRC in "${SRCS[@]}"; do
#     for LANG in $SRC $TGT; do
#         if [ "$LANG" == "$SRC" ]; then
#             t="src"
#         else
#             t="ref"
#         fi
#         grep '<seg id' $ORIG/test-full/newstest2014-${SRC}${TGT}-$t.$LANG.sgm | \
#             sed -e 's/<seg id="[0-9]*">\s*//g' | \
#             sed -e 's/\s*<\/seg>\s*//g' | \
#             sed -e "s/\’/\'/g" | \
#         perl $TOKENIZER -threads 8 -a -LANG $LANG > $TMP/test.${SRC}-${TGT}.$LANG
#         echo ""
#     done
# done


# echo "splitting train and valid..."
# for SRC in "${SRCS[@]}"; do
#     for LANG in $SRC $TGT; do
#         TOK=train.tags.${SRC}-${TGT}.tok.$LANG
#         EXT=${SRC}-${TGT}.$LANG
#         awk '{if (NR%1333 == 0)  print $0; }' $TMP/$TOK > $TMP/valid.$EXT
#         awk '{if (NR%1333 != 0)  print $0; }' $TMP/$TOK > $TMP/train.$EXT
#     done
# done


# # for SRC in "${SRCS[@]}"; do
# #     TRAIN=$TMP/train.$SRC-$TGT
# #     BPE_CODE=$DATA/code.$SRC-$TGT
# #     rm -f $TRAIN
# #     for LANG in $SRC $TGT; do
# #         cat $TMP/train.${SRC}-${TGT}.$LANG >> $TRAIN
# #     done
# # done


# TRAIN=$TMP/train.$JOINED_SRCS-$TGT

# rm -f $TRAIN
# for SRC in "${SRCS[@]}"; do
#     for LANG in $SRC $TGT; do
#         cat $TMP/train.${SRC}-${TGT}.$LANG >> $TRAIN
#     done
# done

# BPE_CODE=$DATA/code.$JOINED_SRCS-$TGT
# echo "learn_bpe.py on ${TRAIN}..."
# python $SPM_TRAIN -s $BPE_TOKENS < $TRAIN > $BPE_CODE

# for SRC in "${SRCS[@]}"; do
#     for LANG in $SRC $TGT; do
#         EXT=${SRC}-${TGT}.$LANG
#         for f in train.$EXT valid.$EXT test.$EXT; do
#             echo "apply_bpe.py to ${f}..."
#             python $SPM_ENCODE -c $BPE_CODE < $TMP/$f > $TMP/bpe.$f
#         done
#     done
# done

# for SRC in "${SRCS[@]}"; do
#     EXT=${SRC}-${TGT}
#     perl $CLEAN -ratio 1.5 $TMP/bpe.train.$EXT $SRC $TGT $DATA/train.$EXT 1 250
#     perl $CLEAN -ratio 1.5 $TMP/bpe.valid.$EXT $SRC $TGT $DATA/valid.$EXT 1 250
# done

# for SRC in "${SRCS[@]}"; do
#     for LANG in $SRC $TGT; do
#         EXT=${SRC}-${TGT}.$LANG
#         cp $TMP/bpe.test.$EXT $DATA/test.$EXT
#     done
# done