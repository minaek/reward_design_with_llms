from typing import List, Tuple

import torch

from utils.data import STOP_TOKENS
from models.dialog_model import DialogModel
from coarse_dialogue_acts.corpus import ActCorpus


class CdaRnnModel(DialogModel):
    corpus_ty = ActCorpus

    def write(self, lang_h: torch.Tensor, ctx_h: torch.Tensor, max_words: int, temperature: float, context_mask: List[float],
              stop_tokens: List[str] = STOP_TOKENS, resume: bool = False) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        max_words = 1
        return super().write(lang_h, ctx_h, max_words, temperature, context_mask, stop_tokens, resume)

