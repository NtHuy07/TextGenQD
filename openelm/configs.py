from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseConfig:
    output_dir: str = "logs/"


@dataclass
class ModelConfig(BaseConfig):
    fp16: bool = True
    cuda: bool = True
    gpus: int = 1
    seed: Optional[int] = None
    deterministic: bool = False
    top_p: float = 0.95
    temp: float = 1.1
    gen_max_len: int = 512
    batch_size: int = 10
    model_type: str = "hf"  # Can be "hf", "gemini"
    model_path: str = MISSING  # Can be HF model name or path to local model
    logits_only: bool = False
    do_sample: bool = True
    num_return_sequences: int = 1
    device: str = "cuda:0"
    system_prompt: bool = False


@dataclass
class PromptModelConfig(ModelConfig):
    model_name: str = "prompt"
    model_path: str = "Salesforce/codegen-350M-mono"


@dataclass
class QDConfig(BaseConfig):
    init_steps: int = 10
    total_steps: int = 50
    history_length: int = 1
    save_history: bool = False
    save_snapshot_interval: int = 10
    log_snapshot_dir: str = ""
    seed: Optional[int] = 42
    save_np_rng_state: bool = False
    load_np_rng_state: bool = False
    crossover: bool = False
    crossover_parents: int = 2
    quality_mutation_prob: float = 0.5
    diversity_mutation_prob: float = 0.5
    pool_size = 9
    not_qd: bool = True


@dataclass
class MAPElitesConfig(QDConfig):
    qd_name: str = "mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (3,))


@dataclass
class EnvConfig(BaseConfig):
    timeout: float = 5.0  # Seconds
    # sandbox: bool = False
    # sandbox_server: str = "http://localhost:5000"
    # processes: int = 1
    batch_size: int = 1  # Batch size of MAP-Elites
    env_name: str = MISSING
    debug: bool = False
    seed: Optional[int] = 42
    wandb_log: bool = True
    wandb_name: str = ""
    llm_sampling: bool = False

@dataclass
class QDTextSumConfig(EnvConfig):
    env_name: str = "qdtextsum"
    quality_metric: str = "logit-score" # ai-score, logit-score 
    diversity_metric: str = "logit-score" # ai-score, logit-score
    behavior: list[int] = field(
        default_factory=lambda: [0, 1, 2]
    )
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [1, 15],
            [1, 10],
            [1, 100],
        ]
    )
    num_test_set: int = 100
    dataset: str = "ccdv/cnn_dailymail"
    dataset_ver: str = "3.0.0"

@dataclass
class QDTextConConfig(EnvConfig):
    env_name: str = "qdtextcon"
    quality_metric: str = "logit-score" # ai-score, logit-score 
    diversity_metric: str = "logit-score" # ai-score, logit-score
    behavior: list[int] = field(
        default_factory=lambda: [0, 1, 2]
    )
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [1, 100],
            [1, 10],
            [1, 100],
        ]
    )
    num_test_set: int = 100
    dataset: str = "stanfordnlp/imdb"
    dataset_ver: str = "plain_text"


@dataclass
class QDCommonGenConfig(EnvConfig):
    env_name: str = "qdcommongen"
    quality_metric: str = "logit-score" # ai-score, logit-score 
    diversity_metric: str = "logit-score" # ai-score, logit-score
    behavior: list[int] = field(
        default_factory=lambda: [0, 1, 2]
    )
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [1, 100],
            [1, 10],
            [1, 100],
        ]
    )
    num_test_set: int = 100
    dataset: str = "GEM/common_gen"
    dataset_ver: str = "default"

@dataclass
class QDData2TextConfig(EnvConfig):
    env_name: str = "qddata2text"
    quality_metric: str = "logit-score" # ai-score, logit-score 
    diversity_metric: str = "logit-score" # ai-score, logit-score
    behavior: list[int] = field(
        default_factory=lambda: [0, 1, 2]
    )
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [1, 100],
            [1, 10],
            [1, 100],
        ]
    )
    num_test_set: int = 20
    dataset: str = "GEM/totto"
    dataset_ver: str = "totto"

@dataclass
class QDQAConfig(EnvConfig):
    env_name: str = "qdqa"
    quality_metric: str = "logit-score" # ai-score, logit-score 
    diversity_metric: str = "logit-score" # ai-score, logit-score
    behavior: list[int] = field(
        default_factory=lambda: [0, 1, 2]
    )
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [1, 100],
            [1, 10],
            [1, 100],
        ]
    )
    num_test_set: int = 100
    dataset: str = "narrativeqa"
    dataset_ver: str = "default"

@dataclass
class QDDialogConfig(EnvConfig):
    env_name: str = "qddialog"
    quality_metric: str = "logit-score" # ai-score, logit-score 
    diversity_metric: str = "logit-score" # ai-score, logit-score
    behavior: list[int] = field(
        default_factory=lambda: [0, 1, 2]
    )
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [1, 100],
            [1, 10],
            [1, 100],
        ]
    )
    num_test_set: int = 100
    dataset: str = "daily_dialog"
    dataset_ver: str = "default"


defaults_elm = [
    {"model": "prompt"},
    {"qd": "mapelites"},
    {"env": "qdtextsum"},
    "_self_",
]


@dataclass
class ELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "logs/elm/${hydra.job.override_dirname}/${now:%y-%m-%d_%H:%M}"
            }
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_elm)
    model: Any = MISSING
    qd: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None




def register_configstore() -> ConfigStore:
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="env", name="qdtextsum", node=QDTextSumConfig)
    cs.store(group="env", name="qdtextcon", node=QDTextConConfig)
    cs.store(group="env", name="qdcommongen", node=QDCommonGenConfig)
    cs.store(group="env", name="qddata2text", node=QDData2TextConfig)
    cs.store(group="env", name="qdqa", node=QDQAConfig)
    cs.store(group="env", name="qddialog", node=QDDialogConfig)

    cs.store(group="qd", name="mapelites", node=MAPElitesConfig)
    cs.store(group="qd", name="ga", node=MAPElitesConfig)
    cs.store(group="model", name="prompt", node=PromptModelConfig)
    cs.store(name="elmconfig", node=ELMConfig)
    return cs


CONFIGSTORE = register_configstore()
