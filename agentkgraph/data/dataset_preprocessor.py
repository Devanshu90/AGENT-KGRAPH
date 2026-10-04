import json
import re
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime,timezone

import pandas as pd

from agentkgraph.data.preprocess_config import (
    PREPROCESSING_VERSION,
    DATASETS,
    METAQA_HOPS,
    HOTPOTQA_SPLITS,
    WEBQSP_SPLITS
)


RAW_DIR=Path("data/raw")
OUT_DIR=Path("data/processed")


def clean_text(text):
    text=str(text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()


def normalize_text(text):
    return clean_text(text).lower()


def make_id(prefix,text):
    h=hashlib.sha1(
        normalize_text(text).encode("utf-8")
    ).hexdigest()[:12]

    return f"{prefix}_{h}"


def write_jsonl(path,records):
    path=Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )+"\n"
            )


def read_metaqa_qa(z,hop,split):
    name=f"{hop}-hop/vanilla/qa_{split}.txt"

    if name not in z.namelist():
        return []

    records=[]

    with z.open(name) as f:
        for i,line in enumerate(f):
            line=line.decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            if "\t" not in line:
                continue

            question,answers=line.split(
                "\t",
                1
            )

            answers=[
                clean_text(x)
                for x in answers.split("|")
                if clean_text(x)
            ]

            records.append({
                "question_id":(
                    f"metaqa_{hop}hop_"
                    f"{split}_{i:07d}"
                ),
                "dataset":"MetaQA",
                "split":split,
                "hop":hop,
                "question":clean_text(
                    question
                ),
                "answers":answers
            })

    return records


def process_metaqa():
    base=RAW_DIR/"metaqa"/"MetaQA"

    result_dir=OUT_DIR/"metaqa"

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    triples=[]

    kb=base/"kb.txt"

    if kb.exists():
        with open(
            kb,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
            for i,line in enumerate(f):
                line=line.strip()

                if not line:
                    continue

                parts=line.split("|")

                if len(parts)<3:
                    continue

                subject=clean_text(
                    parts[0]
                )

                predicate=clean_text(
                    parts[1]
                )

                obj=clean_text(
                    "|".join(parts[2:])
                )

                triples.append({
                    "triple_id":
                        f"metaqa_t_{i:08d}",
                    "dataset":
                        "MetaQA",
                    "subject":
                        subject,
                    "predicate":
                        predicate,
                    "object":
                        obj
                })

    write_jsonl(
        result_dir/"kg_triples.jsonl",
        triples
    )

    qa_by_split={
        "train":[],
        "dev":[],
        "test":[]
    }

    archives=sorted(
        base.glob("*-hop-*.zip")
    )

    for archive in archives:
        name=archive.name.lower()

        if "1-hop" in name:
            hop=1
        elif "2-hop" in name:
            hop=2
        elif "3-hop" in name:
            hop=3
        else:
            continue

        with zipfile.ZipFile(
            archive
        ) as z:
            for split in qa_by_split:
                records=read_metaqa_qa(
                    z,
                    hop,
                    split
                )

                qa_by_split[
                    split
                ].extend(records)

    for split,records in qa_by_split.items():
        write_jsonl(
            result_dir/f"{split}.jsonl",
            records
        )

    return triples,qa_by_split


def process_hotpotqa():
    base=RAW_DIR/"hotpotqa"/"HotpotQA"

    result_dir=OUT_DIR/"hotpotqa"

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    corpus=[]
    qa_by_split={}

    for split in HOTPOTQA_SPLITS:
        files=list(
            base.glob(
                f"**/{split}-*.parquet"
            )
        )

        if not files:
            continue

        records=[]

        for parquet_file in files:
            df=pd.read_parquet(
                parquet_file
            )

            for _,row in df.iterrows():

                question=clean_text(
                    row["question"]
                )

                answer=clean_text(
                    row["answer"]
                )

                qid=str(row["id"])

                supporting=set()

                for item in row[
                    "supporting_facts"
                ]:
                    if (
                        isinstance(
                            item,
                            (list,tuple)
                        )
                        and len(item)>=2
                    ):
                        supporting.add(
                            (
                                str(item[0]),
                                int(item[1])
                            )
                        )

                contexts=row["context"]

                if isinstance(
                    contexts,
                    dict
                ):
                    contexts=list(
                        contexts.items()
                    )

                for context_item in contexts:

                    if isinstance(
                        context_item,
                        dict
                    ):
                        title=context_item.get(
                            "title",
                            ""
                        )

                        sentences=context_item.get(
                            "sentences",
                            []
                        )

                    else:
                        title=context_item[0]
                        sentences=context_item[1]

                    title=clean_text(
                        title
                    )

                    for sentence_id,sentence in enumerate(
                        sentences
                    ):
                        sentence=clean_text(
                            sentence
                        )

                        if not sentence:
                            continue

                        passage_id=make_id(
                            "hotpotqa",
                            (
                                f"{qid}|"
                                f"{title}|"
                                f"{sentence_id}|"
                                f"{sentence}"
                            )
                        )

                        is_supporting=(
                            title,
                            sentence_id
                        ) in supporting

                        corpus.append({
                            "passage_id":
                                passage_id,
                            "dataset":
                                "HotpotQA",
                            "document_id":
                                qid,
                            "title":
                                title,
                            "sentence_id":
                                sentence_id,
                            "text":
                                sentence,
                            "metadata":{
                                "supporting_fact":
                                    is_supporting
                            }
                        })

                records.append({
                    "question_id":
                        qid,
                    "dataset":
                        "HotpotQA",
                    "split":
                        split,
                    "question":
                        question,
                    "answer":
                        answer,
                    "type":
                        str(row["type"]),
                    "level":
                        str(row["level"]),
                    "supporting_facts":[
                        {
                            "title":
                                title,
                            "sentence_id":
                                sentence_id
                        }
                        for title,sentence_id
                        in sorted(supporting)
                    ]
                })

        qa_by_split[
            split
        ]=records

        write_jsonl(
            result_dir/f"{split}.jsonl",
            records
        )

    return corpus,qa_by_split


def load_webqsp_file(z,path):
    return json.loads(
        z.read(path).decode(
            "utf-8"
        )
    )


def process_webqsp():
    archive=(
        RAW_DIR/
        "webqsp"/
        "WebQSP"/
        "WebQSP.zip"
    )

    result_dir=OUT_DIR/"webqsp"

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    qa_by_split={}

    with zipfile.ZipFile(
        archive
    ) as z:

        for split in WEBQSP_SPLITS:

            path=(
                f"WebQSP/data/"
                f"WebQSP.{split}.json"
            )

            if path not in z.namelist():
                print(
                    f"  Warning: {path} not found"
                )
                continue

            data=load_webqsp_file(
                z,
                path
            )

            records=[]

            for question in data.get(
                "Questions",
                []
            ):

                parses=[]

                for parse in question.get(
                    "Parses",
                    []
                ):

                    answers=[]

                    for answer in parse.get(
                        "Answers",
                        []
                    ):
                        answers.append({
                            "type":
                                answer.get(
                                    "AnswerType"
                                ),
                            "argument":
                                answer.get(
                                    "AnswerArgument"
                                ),
                            "name":
                                answer.get(
                                    "EntityName"
                                )
                        })

                    parses.append({
                        "parse_id":
                            parse.get(
                                "ParseId"
                            ),
                        "sparql":
                            parse.get(
                                "Sparql"
                            ),
                        "topic_entity_mention":
                            parse.get(
                                "PotentialTopicEntityMention"
                            ),
                        "topic_entity_name":
                            parse.get(
                                "TopicEntityName"
                            ),
                        "topic_entity_mid":
                            parse.get(
                                "TopicEntityMid"
                            ),
                        "inferential_chain":
                            parse.get(
                                "InferentialChain",
                                []
                            ),
                        "constraints":
                            parse.get(
                                "Constraints",
                                []
                            ),
                        "time":
                            parse.get(
                                "Time"
                            ),
                        "order":
                            parse.get(
                                "Order"
                            ),
                        "answers":
                            answers
                    })

                records.append({
                    "question_id":
                        question.get(
                            "QuestionId"
                        ),
                    "dataset":
                        "WebQSP",
                    "split":
                        split,
                    "question":
                        clean_text(
                            question.get(
                                "RawQuestion",
                                ""
                            )
                        ),
                    "processed_question":
                        clean_text(
                            question.get(
                                "ProcessedQuestion",
                                ""
                            )
                        ),
                    "parses":
                        parses
                })

            write_jsonl(
                result_dir/f"{split}.jsonl",
                records
            )

            qa_by_split[
                split
            ]=records

    return qa_by_split


def deduplicate_corpus(records):
    seen=set()
    result=[]

    for record in records:

        key=normalize_text(
            record["text"]
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        result.append(record)

    return result


def build_corpus(hotpot_corpus):
    corpus=deduplicate_corpus(
        hotpot_corpus
    )

    write_jsonl(
        OUT_DIR/
        "corpus"/
        "passages.jsonl",
        corpus
    )

    return corpus


def save_preprocessing_metadata(
    metaqa_triples,
    metaqa_qa,
    hotpot_corpus,
    hotpot_qa,
    corpus,
    webqsp_qa
):
    metadata={
        "preprocessing_version":
            PREPROCESSING_VERSION,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "datasets":
            DATASETS,

        "configuration":{
            "metaqa_hops":
                METAQA_HOPS,

            "hotpotqa_splits":
                HOTPOTQA_SPLITS,

            "webqsp_splits":
                WEBQSP_SPLITS,

            "corpus_source":
                "HotpotQA",

            "deduplication":
                "normalized passage text"
        },

        "statistics":{
            "metaqa":{
                "kg_triples":
                    len(metaqa_triples),

                "train":
                    len(
                        metaqa_qa.get(
                            "train",
                            []
                        )
                    ),

                "dev":
                    len(
                        metaqa_qa.get(
                            "dev",
                            []
                        )
                    ),

                "test":
                    len(
                        metaqa_qa.get(
                            "test",
                            []
                        )
                    )
            },

            "hotpotqa":{
                "train":
                    len(
                        hotpot_qa.get(
                            "train",
                            []
                        )
                    ),

                "validation":
                    len(
                        hotpot_qa.get(
                            "validation",
                            []
                        )
                    ),

                "test":
                    len(
                        hotpot_qa.get(
                            "test",
                            []
                        )
                    ),

                "raw_passages":
                    len(hotpot_corpus),

                "unique_passages":
                    len(corpus)
            },

            "webqsp":{
                "train":
                    len(
                        webqsp_qa.get(
                            "train",
                            []
                        )
                    ),

                "test":
                    len(
                        webqsp_qa.get(
                            "test",
                            []
                        )
                    )
            }
        }
    }

    path=(
        OUT_DIR/
        "preprocessing_metadata.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    return path


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Processing MetaQA..."
    )

    metaqa_triples,metaqa_qa=(
        process_metaqa()
    )

    print(
        f"  KG triples: "
        f"{len(metaqa_triples)}"
    )

    print(
        "  QA:",
        {
            k:len(v)
            for k,v
            in metaqa_qa.items()
        }
    )

    print(
        "Processing HotpotQA..."
    )

    hotpot_corpus,hotpot_qa=(
        process_hotpotqa()
    )

    print(
        "  QA:",
        {
            k:len(v)
            for k,v
            in hotpot_qa.items()
        }
    )

    print(
        f"  passages: "
        f"{len(hotpot_corpus)}"
    )

    print(
        "Processing WebQSP..."
    )

    webqsp_qa=process_webqsp()

    print(
        "  QA:",
        {
            k:len(v)
            for k,v
            in webqsp_qa.items()
        }
    )

    print(
        "Building unified corpus..."
    )

    corpus=build_corpus(
        hotpot_corpus
    )

    print(
        f"  unique passages: "
        f"{len(corpus)}"
    )

    metadata_path=(
        save_preprocessing_metadata(
            metaqa_triples,
            metaqa_qa,
            hotpot_corpus,
            hotpot_qa,
            corpus,
            webqsp_qa
        )
    )

    print(
        f"  metadata: "
        f"{metadata_path}"
    )

    print(
        "\nPreprocessing complete."
    )


if __name__=="__main__":
    main()