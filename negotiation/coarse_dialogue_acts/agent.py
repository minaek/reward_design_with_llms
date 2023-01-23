from typing import List
from utils.agent import RlAgent


class CdaAgent(RlAgent):
    def write(self, max_words=1) -> List[str]:  # Cap response length to just be the CDA
        return super().write(max_words=max_words)
