from dataclasses import dataclass,field


@dataclass
class KGConfig:
    hop_limit:int=2
    top_paths:int=8
    path_min_score:float=0.10
    decay_rate:float=0.02
    confidence_floor:float=0.15
    commit_min_conf:float=0.60
    max_seeds:int=5


@dataclass
class RetrievalConfig:
    top_k:int=4
    hybrid_alpha:float=0.5


@dataclass
class RoutingConfig:
    actions:list=field(
        default_factory=lambda:[
            "vector",
            "kg",
            "hybrid"
        ]
    )
    ucb_beta:float=1.0
    min_explore:int=30


@dataclass
class EvolutionConfig:
    n:int=2
    threshold:float=0.80
    max_retry:int=2
    user_feedback_weight:float=0.30


@dataclass
class ModelConfig:
    extractor:str="Qwen/Qwen2.5-1.5B-Instruct"
    extractor_adapter:str|None=None
    verifier:str="cross-encoder/nli-deberta-v3-base"
    router:str="Qwen/Qwen2.5-0.5B-Instruct"
    synthesizer:str="Qwen/Qwen2.5-3B-Instruct"
    embedder:str="BAAI/bge-small-en-v1.5"
    extractor_max_new_tokens:int=256
    extractor_temperature:float=0.1
    router_max_new_tokens:int=80
    synthesizer_max_new_tokens:int=128
    synthesizer_temperature:float=0.2
    load_in_4bit:bool=False


@dataclass
class Config:
    kg:KGConfig=field(default_factory=KGConfig)
    retrieval:RetrievalConfig=field(default_factory=RetrievalConfig)
    routing:RoutingConfig=field(default_factory=RoutingConfig)
    evolution:EvolutionConfig=field(default_factory=EvolutionConfig)
    models:ModelConfig=field(default_factory=ModelConfig)
