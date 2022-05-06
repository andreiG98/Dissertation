#!/bin/bash

BPE_TOKENS=16000

while [ $# -gt 0 ] ; do
  case $1 in
    -s | --source) SRC="$2" ;;
    -t | --target) TGT="$2" ;;
    --vocab_size) VS="$2" ;;
  esac
  shift
done

if [ ! -z "$VS" ]; then
    BPE_TOKENS=$VS
fi

echo $BPE_TOKENS

# while getopts s:t: flag
# do
#     case "${flag}" in
#         s) SRC=${OPTARG};;
#         t) TGT=${OPTARG};;
#     esac
# done

cd ../tools/
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git
cd ../scripts

ROOT=$(dirname "$0")
SCRIPTS=../tools/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=../tools/subword-nmt/subword_nmt
SPM_TRAIN=$BPEROOT/learn_bpe.py
SPM_ENCODE=$BPEROOT/apply_bpe.py
FAIRSEQ_SCRIPTS=fairseq_scripts
FAIRSEQ_SPM_TRAIN=$FAIRSEQ_SCRIPTS/spm_train.py
FAIRSEQ_SPM_ENCODE=$FAIRSEQ_SCRIPTS/spm_encode.py

CORPORA=(
    # "MultiParaCrawl"
    # "TED2020"
    # "WikiMatrix"
    "Europarl"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

SRCS=(
    $SRC
)
TGT=$TGT
DATA=$ROOT/../data
TMP=$DATA/tmp
ORIG=$DATA/orig
PROCESSED=$DATA/processed
TMP_DOWNLOADS=$ORIG/tmp_downloads
SPM_FOLDER=$DATA/spm_models

mkdir -p $TMP_DOWNLOADS $PROCESSED $ORIG $TMP $DATA $SPM_FOLDER

cd $ORIG

for SRC in "${SRCS[@]}"; do
    # download and extract data
    DATA_FOLDER=$SRC-$TGT
    if [ -d "$DATA_FOLDER" ]; then
        echo "$DATA_FOLDER already exists, skipping download"
    else
        mkdir -p $DATA_FOLDER
        opus_express -s $SRC -t $TGT -c ${CORPORA[*]} -q \
            --download-dir tmp_downloads/ \
            --test-set $DATA_FOLDER/test \
            --dev-set $DATA_FOLDER/dev \
            --train-set $DATA_FOLDER/train
    fi
done

rm -rf $TMP_DOWNLOADS

cd ../../scripts

echo $PWD

echo "pre-processing train/dev data..."
for SRC in "${SRCS[@]}"; do
    for LANG in $SRC $TGT; do
        DATA_FOLDER=${SRC}-${TGT}

        mkdir -p $TMP/$DATA_FOLDER
        mkdir -p $TMP/$DATA_FOLDER/$BPE_TOKENS
        
        for phase in "train" "dev"; do
            TOK=$DATA_FOLDER/$phase.tags.tok.$LANG
            rm $TMP/$TOK
            cat $ORIG/$DATA_FOLDER/$phase.$LANG | \
                perl $NORM_PUNC $LANG | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 8 -a -l $LANG | \
                perl $LC >> $TMP/$TOK
        done
    done
done

echo "cleaning train/dev data..."
for SRC in "${SRCS[@]}"; do
    DATA_FOLDER=${SRC}-${TGT}
    for phase in "train" "dev"; do
        TOK=$DATA_FOLDER/$phase.tags.tok
        TOK_CLEAN=$DATA_FOLDER/$phase.tags.tok.clean

        perl $CLEAN -ratio 1.5 $TMP/$TOK $SRC $TGT $TMP/$TOK_CLEAN 1 250
    done
done

echo "pre-processing test data..."
for SRC in "${SRCS[@]}"; do
    for LANG in $SRC $TGT; do
        DATA_FOLDER=${SRC}-${TGT}
        
        TOK=$DATA_FOLDER/test.tags.tok.$LANG
        rm $TMP/$TOK
        cat $ORIG/$DATA_FOLDER/test.$LANG | \
            perl $TOKENIZER -threads 8 -a -l $LANG >> $TMP/$TOK
    done
done


# learn BPE with sentencepiece
TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do DATA_FOLDER=${SRC}-${TGT}; echo $TMP/$DATA_FOLDER/train.tags.tok.clean.$SRC ; echo $TMP/$DATA_FOLDER/train.tags.tok.clean.$TGT ; done | tr "\n" ",")
echo "learning joint BPE over ${TRAIN_FILES}..."
python3 "$FAIRSEQ_SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$SPM_FOLDER/sentencepiece_$SRC-$TGT_$BPE_TOKENS.bpe \
    --vocab_size=$BPE_TOKENS \
    --character_coverage=1.0 \
    --model_type=bpe

echo "encoding train/dev/test with learned BPE..."
for SRC in "${SRCS[@]}"; do
    DATA_FOLDER=${SRC}-${TGT}
    DATA_FOLDER_BPE=${DATA_FOLDER}/${BPE_TOKENS}

    for LANG in $SRC $TGT; do
        for phase in "train" "dev" "test"; do
            TOK=$DATA_FOLDER/$phase.tags.tok.clean.$LANG
            [ "$phase" == "test" ] && TOK=$DATA_FOLDER/$phase.tags.tok.$LANG
            echo "$TOK"
            BPE_FILE=$DATA_FOLDER_BPE/$phase.bpe.$LANG
            python3 "$FAIRSEQ_SPM_ENCODE" \
                --model $SPM_FOLDER/sentencepiece_$SRC-$TGT_$BPE_TOKENS.bpe.model \
                --output_format=piece \
                --inputs $TMP/$TOK \
                --outputs $TMP/$BPE_FILE
                # --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
        done
    done
done

echo "copying train/dev/test to data folder"
for SRC in "${SRCS[@]}"; do
    DATA_FOLDER=${SRC}-${TGT}
    DATA_FOLDER_BPE=${DATA_FOLDER}/${BPE_TOKENS}
    mkdir -p $PROCESSED/$DATA_FOLDER
    mkdir -p $PROCESSED/$DATA_FOLDER_BPE

    for LANG in $SRC $TGT; do
        for phase in "train" "dev" "test"; do
            BPE_FILE=$DATA_FOLDER_BPE/$phase.bpe.$LANG
            cp $TMP/$BPE_FILE $PROCESSED/$BPE_FILE
        done
    done
done