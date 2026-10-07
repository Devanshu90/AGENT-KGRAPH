import json
from pathlib import Path


class EvaluationReport:
    def __init__(self,title="AGENT-KGRAPH Evaluation Report"):
        self.title=title
        self.sections=[]

    def add_section(self,name,data):
        self.sections.append({
            "name":name,
            "data":dict(data or {}),
        })

    def to_dict(self):
        return {
            "title":self.title,
            "sections":self.sections,
        }

    def save_json(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)

        with path.open("w",encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=2,
            )

        return path

    def to_text(self):
        lines=[
            self.title,
            "="*len(self.title),
            "",
        ]

        for section in self.sections:
            lines.append(section["name"])
            lines.append("-"*len(section["name"]))

            for key,value in section["data"].items():
                if isinstance(value,(dict,list)):
                    value=json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                lines.append(
                    f"{key}: {value}"
                )

            lines.append("")

        return "\n".join(lines)

    def save_text(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)

        with path.open("w",encoding="utf-8") as f:
            f.write(self.to_text())

        return path