from typing import Dict

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class OfflineImageRunner(BaseImageRunner):
    """
    Dummy runner for purely offline training with no environment rollout.

    It simply returns an empty metrics dict, so training can proceed without
    requiring a simulator.
    """

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}

