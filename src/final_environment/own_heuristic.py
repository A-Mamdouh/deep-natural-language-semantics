from typing import List, Optional
from torch import nn
import torch

from dataclasses import dataclass

from src.final_environment.model_generation import (
    Dialog,
    ModelTreeNode,
    get_unique_models,
)
from src.final_environment.natural_language import Sentence, tableau_to_sentence
from src.heuristics.learned_heuristics.deep_learning_models.word_encoder import (
    WordEncoder,
)


@dataclass
class Config:
    n_ctx_tokens: int = 10
    n_attention_heads: int = 7
    compress_size: int = 1 << 8  # 256
    device: str = "cuda"


class NNModel(nn.Module):
    def __init__(
        self, cfg: Optional[Config] = None, word_encoder: Optional[WordEncoder] = None
    ):
        super().__init__()
        self.cfg = cfg or Config()
        self.word_encoder = word_encoder or WordEncoder(device=self.cfg.device)
        self.param_evt = nn.Parameter(torch.rand(self.cfg.compress_size))
        self.param_sub = nn.Parameter(torch.rand(self.cfg.compress_size))
        self.param_vrb = nn.Parameter(torch.rand(self.cfg.compress_size))
        self.param_obj = nn.Parameter(torch.rand(self.cfg.compress_size))
        self.param_adj = nn.Parameter(torch.rand(self.cfg.compress_size))
        self.param_scr = nn.Parameter(torch.rand(self.cfg.compress_size))
        self.param_ctx = nn.Parameter(
            torch.rand((self.cfg.n_ctx_tokens, self.cfg.compress_size))
        )
        self.compress = nn.Linear(self.word_encoder.output_size, self.cfg.compress_size)
        self.mha = nn.MultiheadAttention(self.cfg.compress_size, 4)
        self.score_mlp = nn.Linear(self.cfg.compress_size, 1)
        self.to(device=self.cfg.device)

    def _get_word_embedding(self, word: Optional[str]) -> torch.Tensor | int:
        if word:
            return self.word_encoder.encode_word(word).to(device=self.cfg.device)
        return torch.zeros((self.word_encoder.output_size), device=self.cfg.device)

    def embedd_sentence(
        self, sentence: Sentence, ctx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        subject = self._get_word_embedding(sentence.subject.name)
        verb = self._get_word_embedding(sentence.verb)
        object_ = self._get_word_embedding(sentence.object_ and sentence.object_.name)
        adjectives = []
        for adj, noun in sentence.adjectives:
            adjective = self._get_word_embedding(adj)
            if noun.name == sentence.subject.name:
                adjective += subject
            elif sentence.object_ and noun.name == sentence.object_.name:
                adjective += object_
            else:
                adjective += self._get_word_embedding(noun.name)
            adjectives.append(adjective)
        sequence = torch.stack(
            [subject, verb, object_, *adjectives],
            dim=0,
        )
        compressed_sequence = self.compress(sequence)
        subject, verb, object_, *adjectives = compressed_sequence
        subject = subject + self.param_sub
        verb = verb + self.param_vrb
        object_ = object_ + self.param_obj
        adjectives = [adj + self.param_adj for adj in adjectives]
        if ctx is None:
            ctx = self.param_ctx
        mha_input = torch.stack(
            [
                subject,
                verb,
                object_,
                *adjectives,
                self.param_evt,
                *ctx,
                self.param_scr,
            ],
            dim=0,
        )
        return self.mha.forward(mha_input, mha_input, mha_input)[0]

    def order_models(self, dialog: Dialog) -> List[ModelTreeNode]:
        models = dialog.get_models()
        models = get_unique_models(models)
        scores: List[float] = []
        for model in models:
            score_token = ctx_tokens = None
            for sentence in model.to_sentences():
                out = self.embedd_sentence(sentence, ctx_tokens)
                *ctx_tokens, score_token = out[..., -(self.cfg.n_ctx_tokens + 1) :, :]
            scores.append(self.score_mlp(score_token))
        output_order = torch.argsort(torch.tensor(scores), descending=True)
        return [models[i] for i in output_order]
