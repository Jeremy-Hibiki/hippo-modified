from dataclasses import asdict

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseMetric:
    global_config: BaseConfig
    metric_name = "base"

    def __init__(self, global_config: BaseConfig | None = None) -> None:
        if global_config is None:
            logger.info("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        logger.info(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")

    def calculate_metric_scores(self) -> tuple[dict[str, float], list]:
        """
        Calculate the total score under this metric and score for each individual example in the input.


        Returns:
            Tuple[Dict[str, Union[int, float]], List[Union[int, float]]]
        """
        return {}, []
