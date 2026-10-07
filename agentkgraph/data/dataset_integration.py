import json
from dataclasses import dataclass,field
from pathlib import Path


@dataclass
class DatasetRecord:
    dataset:str
    split:str
    question_id:str
    question:str
    answers:list=field(default_factory=list)
    hop:int|None=None
    supporting_facts:list=field(default_factory=list)
    passage_ids:list=field(default_factory=list)
    parses:list=field(default_factory=list)
    metadata:dict=field(default_factory=dict)


class DatasetIntegration:
    def __init__(self,processed_dir="data/processed"):
        self.processed_dir=Path(processed_dir)

    def _read_jsonl(self,path):
        path=self.processed_dir/path

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        records=[]

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:
            for line_number,line in enumerate(f,1):
                line=line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in {path} "
                        f"at line {line_number}: {e}"
                    )

        return records

    def load_metaqa(self,split):
        if split not in {"train","dev","test"}:
            raise ValueError(
                f"Invalid MetaQA split: {split}"
            )

        records=self._read_jsonl(
            Path("metaqa")/f"{split}.jsonl"
        )

        result=[]

        for record in records:
            result.append(
                DatasetRecord(
                    dataset="MetaQA",
                    split=split,
                    question_id=str(
                        record["question_id"]
                    ),
                    question=record["question"],
                    answers=list(
                        record.get("answers",[])
                    ),
                    hop=int(record["hop"]),
                    metadata={
                        "source_record":record
                    }
                )
            )

        return result

    def load_hotpotqa(self,split):
        if split not in {
            "train",
            "validation",
            "test"
        }:
            raise ValueError(
                f"Invalid HotpotQA split: {split}"
            )

        records=self._read_jsonl(
            Path("hotpotqa")/f"{split}.jsonl"
        )

        result=[]

        for record in records:
            result.append(
                DatasetRecord(
                    dataset="HotpotQA",
                    split=split,
                    question_id=str(
                        record["question_id"]
                    ),
                    question=record["question"],
                    answers=[record["answer"]],
                    supporting_facts=list(
                        record.get(
                            "supporting_facts",
                            []
                        )
                    ),
                    metadata={
                        "type":record.get("type"),
                        "level":record.get("level"),
                        "source_record":record
                    }
                )
            )

        return result

    def load_webqsp(self,split):
        if split not in {"train","test"}:
            raise ValueError(
                f"Invalid WebQSP split: {split}"
            )

        records=self._read_jsonl(
            Path("webqsp")/f"{split}.jsonl"
        )

        result=[]

        for record in records:
            answers=[]

            for parse in record.get(
                "parses",
                []
            ):
                for answer in parse.get(
                    "answers",
                    []
                ):
                    name=answer.get("name")
                    argument=answer.get("argument")

                    value=name or argument

                    if value is not None:
                        answers.append(
                            str(value)
                        )

            answers=list(
                dict.fromkeys(answers)
            )

            result.append(
                DatasetRecord(
                    dataset="WebQSP",
                    split=split,
                    question_id=str(
                        record["question_id"]
                    ),
                    question=record["question"],
                    answers=answers,
                    parses=list(
                        record.get("parses",[])
                    ),
                    metadata={
                        "processed_question":
                            record.get(
                                "processed_question",
                                ""
                            ),
                        "source_record":record
                    }
                )
            )

        return result

    def load(self,dataset,split):
        normalized=dataset.strip().lower()

        if normalized=="metaqa":
            return self.load_metaqa(split)

        if normalized=="hotpotqa":
            return self.load_hotpotqa(split)

        if normalized=="webqsp":
            return self.load_webqsp(split)

        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    def load_all(self):
        return {
            "MetaQA":{
                split:self.load_metaqa(split)
                for split in (
                    "train",
                    "dev",
                    "test"
                )
            },
            "HotpotQA":{
                split:self.load_hotpotqa(split)
                for split in (
                    "train",
                    "validation",
                    "test"
                )
            },
            "WebQSP":{
                split:self.load_webqsp(split)
                for split in (
                    "train",
                    "test"
                )
            }
        }

    def load_metaqa_kg(self):
        return self._read_jsonl(
            Path("metaqa")/
            "kg_triples.jsonl"
        )

    def load_corpus(self):
        return self._read_jsonl(
            Path("corpus")/
            "passages.jsonl"
        )

    def summary(self):
        datasets=self.load_all()

        return {
            dataset:{
                split:len(records)
                for split,records
                in splits.items()
            }
            for dataset,splits
            in datasets.items()
        }