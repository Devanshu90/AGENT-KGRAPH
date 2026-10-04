import json
from pathlib import Path


OUT_DIR=Path("data/processed")


def read_jsonl(path):
    records=[]

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line_number,line in enumerate(f,1):
            line=line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {e}"
                )

    return records


def check_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )


def check_fields(records,required,name):
    for i,record in enumerate(records):
        missing=[
            field
            for field in required
            if field not in record
        ]

        if missing:
            raise ValueError(
                f"{name}: record {i} "
                f"missing fields: {missing}"
            )


def check_unique_ids(records,field,name):
    ids=[
        record.get(field)
        for record in records
        if record.get(field) is not None
    ]

    duplicates=[]
    seen=set()

    for value in ids:
        if value in seen:
            duplicates.append(value)
        else:
            seen.add(value)

    if duplicates:
        print(
            f"  Warning: {name} contains "
            f"{len(set(duplicates))} duplicate "
            f"{field} values"
        )

        print(
            f"  This is allowed for this dataset."
        )


def check_nonempty(value,name):
    if not value:
        raise ValueError(
            f"{name}: empty value"
        )


def validate_metaqa():
    base=OUT_DIR/"metaqa"

    files={
        "triples":base/"kg_triples.jsonl",
        "train":base/"train.jsonl",
        "dev":base/"dev.jsonl",
        "test":base/"test.jsonl"
    }

    for path in files.values():
        check_file(path)

    triples=read_jsonl(
        files["triples"]
    )

    check_fields(
        triples,
        [
            "triple_id",
            "dataset",
            "subject",
            "predicate",
            "object"
        ],
        "MetaQA triples"
    )

    check_unique_ids(
        triples,
        "triple_id",
        "MetaQA triples"
    )

    qa_counts={}

    for split in [
        "train",
        "dev",
        "test"
    ]:
        records=read_jsonl(
            files[split]
        )

        check_fields(
            records,
            [
                "question_id",
                "dataset",
                "split",
                "hop",
                "question",
                "answers"
            ],
            f"MetaQA {split}"
        )

        check_unique_ids(
            records,
            "question_id",
            f"MetaQA {split}"
        )

        for record in records:
            check_nonempty(
                record["question"],
                f"MetaQA {split} question"
            )

            if not isinstance(
                record["answers"],
                list
            ):
                raise ValueError(
                    f"MetaQA {split}: "
                    f"answers is not a list"
                )

            if record["hop"] not in [
                1,
                2,
                3
            ]:
                raise ValueError(
                    f"MetaQA {split}: "
                    f"invalid hop"
                )

        qa_counts[split]=len(records)

    return {
        "triples":len(triples),
        "qa":qa_counts
    }


def validate_hotpotqa():
    base=OUT_DIR/"hotpotqa"

    files={
        "train":base/"train.jsonl",
        "validation":base/"validation.jsonl",
        "test":base/"test.jsonl"
    }

    for path in files.values():
        check_file(path)

    qa_counts={}

    for split,path in files.items():
        records=read_jsonl(path)

        check_fields(
            records,
            [
                "question_id",
                "dataset",
                "split",
                "question",
                "answer",
                "type",
                "level",
                "supporting_facts"
            ],
            f"HotpotQA {split}"
        )

        check_unique_ids(
            records,
            "question_id",
            f"HotpotQA {split}"
        )

        for record in records:
            check_nonempty(
                record["question"],
                f"HotpotQA {split} question"
            )

            if not isinstance(
                record["supporting_facts"],
                list
            ):
                raise ValueError(
                    f"HotpotQA {split}: "
                    f"supporting_facts "
                    f"is not a list"
                )

        qa_counts[split]=len(records)

    corpus_path=(
        OUT_DIR/
        "corpus"/
        "passages.jsonl"
    )

    check_file(corpus_path)

    passages=read_jsonl(
        corpus_path
    )

    check_fields(
        passages,
        [
            "passage_id",
            "dataset",
            "document_id",
            "title",
            "sentence_id",
            "text",
            "metadata"
        ],
        "HotpotQA corpus"
    )

    check_unique_ids(
        passages,
        "passage_id",
        "HotpotQA corpus"
    )

    for record in passages:
        check_nonempty(
            record["text"],
            "HotpotQA corpus text"
        )

        if not isinstance(
            record["metadata"],
            dict
        ):
            raise ValueError(
                "HotpotQA corpus: "
                "metadata is not a dict"
            )

        if "supporting_fact" not in (
            record["metadata"]
        ):
            raise ValueError(
                "HotpotQA corpus: "
                "missing supporting_fact"
            )

    return {
        "qa":qa_counts,
        "unique_passages":len(passages)
    }


def validate_webqsp():
    base=OUT_DIR/"webqsp"

    files={
        "train":base/"train.jsonl",
        "test":base/"test.jsonl"
    }

    for path in files.values():
        check_file(path)

    counts={}

    for split,path in files.items():
        records=read_jsonl(path)

        check_fields(
            records,
            [
                "question_id",
                "dataset",
                "split",
                "question",
                "processed_question",
                "parses"
            ],
            f"WebQSP {split}"
        )

        check_unique_ids(
            records,
            "question_id",
            f"WebQSP {split}"
        )

        for record in records:
            check_nonempty(
                record["question"],
                f"WebQSP {split} question"
            )

            if not isinstance(
                record["parses"],
                list
            ):
                raise ValueError(
                    f"WebQSP {split}: "
                    f"parses is not a list"
                )

            for parse in record["parses"]:
                required=[
                    "parse_id",
                    "sparql",
                    "inferential_chain",
                    "constraints",
                    "answers"
                ]

                missing=[
                    field
                    for field in required
                    if field not in parse
                ]

                if missing:
                    raise ValueError(
                        f"WebQSP {split}: "
                        f"parse missing "
                        f"{missing}"
                    )

        counts[split]=len(records)

    return counts


def validate_metadata(
    metaqa,
    hotpotqa,
    webqsp
):
    path=(
        OUT_DIR/
        "preprocessing_metadata.json"
    )

    check_file(path)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        metadata=json.load(f)

    if "preprocessing_version" not in metadata:
        raise ValueError(
            "Metadata missing "
            "preprocessing_version"
        )

    if "datasets" not in metadata:
        raise ValueError(
            "Metadata missing datasets"
        )

    statistics=metadata.get(
        "statistics",
        {}
    )

    meta_stats=statistics.get(
        "metaqa",
        {}
    )

    if meta_stats.get(
        "kg_triples"
    ) != metaqa["triples"]:
        raise ValueError(
            "Metadata MetaQA triple "
            "count does not match "
            "actual data"
        )

    for split in [
        "train",
        "dev",
        "test"
    ]:
        if meta_stats.get(
            split
        ) != metaqa["qa"].get(
            split
        ):
            raise ValueError(
                f"Metadata MetaQA "
                f"{split} count does not "
                f"match actual data"
            )

    hot_stats=statistics.get(
        "hotpotqa",
        {}
    )

    for split in [
        "train",
        "validation",
        "test"
    ]:
        if hot_stats.get(
            split
        ) != hotpotqa["qa"].get(
            split
        ):
            raise ValueError(
                f"Metadata HotpotQA "
                f"{split} count does not "
                f"match actual data"
            )

    if hot_stats.get(
        "unique_passages"
    ) != hotpotqa[
        "unique_passages"
    ]:
        raise ValueError(
            "Metadata HotpotQA "
            "passage count does not "
            "match actual data"
        )

    web_stats=statistics.get(
        "webqsp",
        {}
    )

    for split in [
        "train",
        "test"
    ]:
        if web_stats.get(
            split
        ) != webqsp.get(split):
            raise ValueError(
                f"Metadata WebQSP "
                f"{split} count does not "
                f"match actual data"
            )

    return True


def main():

    print("="*60)
    print(
        "AGENT-KGRAPH DATASET VALIDATION"
    )
    print("="*60)

    print("\nValidating MetaQA...")

    metaqa=validate_metaqa()

    print(
        f"  KG triples: "
        f"{metaqa['triples']}"
    )

    print(
        f"  QA counts: "
        f"{metaqa['qa']}"
    )

    print(
        "  MetaQA validation: PASS"
    )

    print("\nValidating HotpotQA...")

    hotpotqa=validate_hotpotqa()

    print(
        f"  QA counts: "
        f"{hotpotqa['qa']}"
    )

    print(
        f"  Unique passages: "
        f"{hotpotqa['unique_passages']}"
    )

    print(
        "  HotpotQA validation: PASS"
    )

    print("\nValidating WebQSP...")

    webqsp=validate_webqsp()

    print(
        f"  QA counts: "
        f"{webqsp}"
    )

    print(
        "  WebQSP validation: PASS"
    )

    print(
        "\nValidating metadata..."
    )

    validate_metadata(
        metaqa,
        hotpotqa,
        webqsp
    )

    print(
        "  Metadata validation: PASS"
    )

    print("\n"+"="*60)
    print(
        "ALL DATASET VALIDATIONS PASSED"
    )
    print("="*60)


if __name__=="__main__":
    main()