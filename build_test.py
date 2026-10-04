import json

from agentkgraph.build_kg import KGBuildPipeline


def main():
    with open(
        "data/processed/kg_test.json",
        "r",
        encoding="utf-8"
    ) as f:
        passages=json.load(f)

    pipeline=KGBuildPipeline()

    count=pipeline.build(passages)

    pipeline.save(
        "data/kg/graph.json"
    )

    print("Triples committed:",count)
    print("Entities:",len(pipeline.kg.graph.nodes))
    print("Relations:",len(pipeline.kg.graph.edges))

    print("\nKnowledge Graph:")

    for u,v,data in pipeline.kg.graph.edges(data=True):
        print(
            f"{u} --{data['predicate']}--> {v}"
        )

    print("\nSaved: data/kg/graph.json")


if __name__=="__main__":
    main()