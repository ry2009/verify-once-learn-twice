from __future__ import annotations

from typing import List, Tuple

import tinker

from .config import ExperimentConfig, SamplingConfig


class TinkerRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.service_client = tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(
            base_model=cfg.base_model,
            rank=cfg.lora_rank,
            seed=cfg.seed,
        )
        self.sampling_client = self.service_client.create_sampling_client(
            base_model=cfg.base_model
        )
        judge_model = cfg.judge_model or cfg.base_model
        self.judge_sampling_client = self.service_client.create_sampling_client(
            base_model=judge_model
        )
        self.tokenizer = self.training_client.get_tokenizer()
        self.judge_tokenizer = self.judge_sampling_client.get_tokenizer()

    def sampling_params(self, sampling_cfg: SamplingConfig) -> tinker.SamplingParams:
        return tinker.SamplingParams(
            max_tokens=sampling_cfg.max_tokens,
            temperature=sampling_cfg.temperature,
            top_p=sampling_cfg.top_p,
            stop=sampling_cfg.stop,
        )

    def _to_model_input(self, text: str, *, judge: bool = False) -> tinker.ModelInput:
        tokenizer = self.judge_tokenizer if judge else self.tokenizer
        return tinker.ModelInput.from_ints(
            tokenizer.encode(text, add_special_tokens=True)
        )

    def sample_one(self, prompt: str, sampling_cfg: SamplingConfig) -> str:
        params = self.sampling_params(sampling_cfg)
        outputs = self.sampling_client.sample(
            prompt=self._to_model_input(prompt),
            num_samples=1,
            sampling_params=params,
        ).result()
        return self.tokenizer.decode(
            outputs.sequences[0].tokens, skip_special_tokens=True
        )

    def sample_one_judge(self, prompt: str, sampling_cfg: SamplingConfig) -> str:
        params = self.sampling_params(sampling_cfg)
        outputs = self.judge_sampling_client.sample(
            prompt=self._to_model_input(prompt, judge=True),
            num_samples=1,
            sampling_params=params,
        ).result()
        return self.judge_tokenizer.decode(
            outputs.sequences[0].tokens, skip_special_tokens=True
        )

    def make_datum(self, prompt: str, completion: str) -> tinker.Datum:
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
        tokens = prompt_tokens + completion_tokens
        if len(tokens) < 2:
            raise ValueError("Not enough tokens to train")
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": weights,
            },
        )

    def train_on_pairs(self, pairs: List[Tuple[str, str]]) -> None:
        data = [self.make_datum(p, c) for p, c in pairs]
        fwd = self.training_client.forward_backward(
            data=data,
            loss_fn="cross_entropy",
        )
        fwd.result()
        optim = self.training_client.optim_step(
            adam_params=tinker.AdamParams(
                learning_rate=self.cfg.lr,
                weight_decay=0.0,
            )
        )
        optim.result()

    def refresh_sampling_client(self, name: str) -> None:
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=name
        )
        self.tokenizer = self.sampling_client.get_tokenizer()
