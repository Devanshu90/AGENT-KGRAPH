from pathlib import Path


RAW_DIR=Path("data/raw")
OUT_DIR=Path("data/processed")

METAQA_DIR=RAW_DIR/"metaqa"/"MetaQA"
HOTPOTQA_DIR=RAW_DIR/"hotpotqa"/"HotpotQA"
WEBQSP_DIR=RAW_DIR/"webqsp"/"WebQSP"

CORPUS_DIR=OUT_DIR/"corpus"

METAQA_OUTPUT_DIR=OUT_DIR/"metaqa"
HOTPOTQA_OUTPUT_DIR=OUT_DIR/"hotpotqa"
WEBQSP_OUTPUT_DIR=OUT_DIR/"webqsp"

PREPROCESSING_VERSION="1.0"

DATASETS=[
    "MetaQA",
    "HotpotQA",
    "WebQSP"
]

METAQA_HOPS=[1,2,3]

HOTPOTQA_SPLITS=[
    "train",
    "validation",
    "test"
]

WEBQSP_SPLITS=[
    "train",
    "test"
]

METAQA_KG_FILE="kb.txt"

WEBQSP_ARCHIVE="WebQSP.zip"

PASSAGE_ID_PREFIX="hotpotqa"

TEXT_NORMALIZATION={
    "whitespace":True,
    "lowercase_for_hash":True,
    "sha1_id_length":12
}