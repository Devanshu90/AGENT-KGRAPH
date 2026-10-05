import re


class Router:
    def __init__(self,config=None):
        self.config=config
        self.actions=[
            "vector",
            "kg",
            "hybrid"
        ]

    def features(self,query):
        q=query.lower()

        return {
            "has_entity":bool(
                re.search(
                    r"\[[^\]]+\]",
                    query
                )
            ),
            "has_who":q.startswith("who"),
            "has_what":q.startswith("what"),
            "has_where":q.startswith("where"),
            "has_when":q.startswith("when"),
            "has_how":q.startswith("how"),
            "length":len(q.split())
        }

    def route(self,query):
        f=self.features(query)

        if f["has_entity"]:
            return "kg"

        if (
            f["has_who"]
            and "direct" in query.lower()
        ):
            return "hybrid"

        if f["has_when"]:
            return "hybrid"

        return "vector"