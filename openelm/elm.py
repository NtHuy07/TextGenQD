from typing import Any, Optional, Type

from hydra.core.hydra_config import HydraConfig

from openelm.configs import DiffModelConfig, ELMConfig, PromptModelConfig
from openelm.mutation_model import get_model
from openelm.environments.base import BaseEnvironment


def load_env(env_name: str) -> Type[BaseEnvironment]:
    if env_name == "qdtextsum":
        from openelm.environments.textsum import TextSumEvolution

        return TextSumEvolution
    
    elif env_name == "qdtextcon":
        from openelm.environments.textcon import TextConEvolution

        return TextConEvolution
    
    elif env_name == "qdcommongen":
        from openelm.environments.commongen import CommonGenEvolution

        return CommonGenEvolution
    
    elif env_name == "qddata2text":
        from openelm.environments.data2text import Data2TextEvolution

        return Data2TextEvolution
    
    elif env_name == "qdqa":
        from openelm.environments.qa import QAEvolution

        return QAEvolution
    
    elif env_name == "qddialog":
        from openelm.environments.dialog import DialogEvolution

        return DialogEvolution
    
    else:
        raise ValueError(f"Unknown environment {env_name}")


def load_algorithm(algorithm_name: str) -> Any:
    if algorithm_name == "mapelites":
        from openelm.algorithms.map_elites import MAPElites

        return MAPElites
    
    elif algorithm_name == "cvtmapelites":
        from openelm.algorithms.map_elites import CVTMAPElites

        return CVTMAPElites
    
    elif algorithm_name == "ga":
        from openelm.algorithms.genetic import GA

        return GA

    elif algorithm_name == "cvtga":
        from openelm.algorithms.genetic import CVTGA

        return CVTGA


class ELM:
    def __init__(self, config: ELMConfig) -> None:
        """
        The main class of ELM.

        This class will load a diff model, an environment, and a QD algorithm
        from the passed config.

        Args:
            config: The config containing the diff model, environment, and QD algorithm.
            env (Optional): An optional environment to pass in. Defaults to None.
        """
        self.config: ELMConfig = config
        self.env_name: str = self.config.env.env_name
        self.qd_name: str = self.config.qd.qd_name
        self.mutation_model = get_model(config=config.model)

        self.environment = load_env(self.env_name)(
            config=self.config.env,
            qd_config=self.config.qd,
            model_config=self.config.model,
            mutation_model=self.mutation_model,
        )

    def run(
        self, init_steps: Optional[int] = None, total_steps: Optional[int] = None, data: Optional[Any] = None, data_id: int = None,
    ) -> str:
        """
        Run the ELM algorithm to evolve the population in the environment.

        Args:
            init_steps: The number of steps to run the initialisation phase.
            total_steps: The number of steps to run the QD algorithm in total,
            including init_steps.

        Returns:
            str: A string representing the maximum fitness genotype. The
            `qd_algorithm` class attribute will be updated.
        """
        self.environment.set_data(data)
        self.qd_algorithm = load_algorithm(self.qd_name)(
            env=self.environment,
            config=self.config.qd,
            env_config=self.config.env,
            data_id=data_id,
        )
        if init_steps is None:
            init_steps = self.config.qd.init_steps
        if total_steps is None:
            total_steps = self.config.qd.total_steps
        return self.qd_algorithm.search(init_steps=init_steps, total_steps=total_steps)