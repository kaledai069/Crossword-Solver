#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn
from transformers import BertConfig, BertModel
from transformers import ElectraConfig, ElectraModel
from transformers import RobertaConfig, RobertaModel
from transformers.optimization import AdamW
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import ElectraTokenizer
from transformers import AlbertModel, AlbertConfig, AlbertTokenizer
from transformers import MobileBertModel, MobileBertConfig, MobileBertTokenizer
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig

import sys
sys.path.append("DPR/dpr/utils")

from data_utils import Tensorizer
from .biencoder import BiEncoder, DistilBertBiEncoder
from .reader import Reader

logger = logging.getLogger(__name__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_bert_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    question_encoder = HFBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )

    fix_ctx_encoder = (
        args.fix_ctx_encoder if hasattr(args, "fix_ctx_encoder") else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_albert_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    question_encoder = HFAlbertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )
    ctx_encoder = HFAlbertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )

    fix_ctx_encoder = (
        args.fix_ctx_encoder if hasattr(args, "fix_ctx_encoder") else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_albert_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_mobilebert_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    question_encoder = HFMobileBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )
    ctx_encoder = HFMobileBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )

    fix_ctx_encoder = (
        args.fix_ctx_encoder if hasattr(args, "fix_ctx_encoder") else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_mobilebert_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_roberta_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    question_encoder = HFRobertaEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )
    ctx_encoder = HFRobertaEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )

    fix_ctx_encoder = (
        args.fix_ctx_encoder if hasattr(args, "fix_ctx_encoder") else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_roberta_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_distilroberta_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    question_encoder = HFDistilRobertaEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )
    ctx_encoder = HFDistilRobertaEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )

    fix_ctx_encoder = (
        args.fix_ctx_encoder if hasattr(args, "fix_ctx_encoder") else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_roberta_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_electra_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    question_encoder = HFElectraEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )
    ctx_encoder = HFElectraEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )

    fix_ctx_encoder = (
        args.fix_ctx_encoder if hasattr(args, "fix_ctx_encoder") else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_electra_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_tinybert_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    question_encoder = HFTinyBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )
    ctx_encoder = HFTinyBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )

    fix_ctx_encoder = (
        args.fix_ctx_encoder if hasattr(args, "fix_ctx_encoder") else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_tinybert_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_distilbert_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    question_encoder = HFDistilBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )
    ctx_encoder = HFDistilBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )

    fix_ctx_encoder = (
        args.fix_ctx_encoder if hasattr(args, "fix_ctx_encoder") else False
    )
    biencoder = DistilBertBiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder = fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_distilbert_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_bert_reader_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        args.pretrained_model_cfg, projection_dim=args.projection_dim, dropout=dropout
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(args)
    return tensorizer, reader, optimizer


def get_bert_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_bert_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return BertTensorizer(tokenizer, args.sequence_length)

def get_albert_tensorizer(args, tokenizer = None):
    if not tokenizer:
        tokenizer = get_albert_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return AlbertTensorizer(tokenizer, args.sequence_length)

def get_mobilebert_tensorizer(args, tokenizer = None):
    if not tokenizer:
        tokenizer = get_mobilebert_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return MobileBertTensorizer(tokenizer, args.sequence_length)


def get_roberta_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_roberta_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return RobertaTensorizer(tokenizer, args.sequence_length)


def get_electra_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_electra_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return ElectraTensorizer(tokenizer, args.sequence_length)

def get_tinybert_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_tinybert_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return TinyBertTensorizer(tokenizer, args.sequence_length)

def get_distilbert_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_distilbert_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return DistilBertTensorizer(tokenizer, args.sequence_length)

def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )

def get_albert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return AlbertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )

def get_mobilebert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return MobileBertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )

def get_electra_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return ElectraTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )

def get_tinybert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return AutoTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )

def get_distilbert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return DistilBertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )

class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs
    ) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(
            cfg_name, config=cfg, project_dim=projection_dim, **kwargs
        )

    def forward(
        self, input_ids: T, token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            outputs = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
        else:
            hidden_states = None
            outputs = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size
    
class HFAlbertEncoder(AlbertModel):
    def __init__(self, config, project_dim: int = 0):
        AlbertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs
    ) -> AlbertModel:
        cfg = AlbertConfig.from_pretrained(cfg_name if cfg_name else "albert-base-v2")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(
            cfg_name, config=cfg, project_dim=projection_dim, **kwargs
        )

    def forward(
        self, input_ids: T, token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            outputs = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
        else:
            hidden_states = None
            outputs = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size
    
class HFMobileBertEncoder(MobileBertModel):
    def __init__(self, config, project_dim: int = 0):
        MobileBertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs
    ) -> MobileBertModel:
        cfg = MobileBertConfig.from_pretrained(cfg_name if cfg_name else "google/mobilebert-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(
            cfg_name, config=cfg, project_dim=projection_dim, **kwargs
        )

    def forward(
        self, input_ids: T, token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            outputs = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
        else:
            hidden_states = None
            outputs = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size
    
class HFTinyBertEncoder(AutoModel):
    def __init__(self, config, project_dim: int = 0):
        AutoModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs
    ) -> AutoModel:
        cfg = AutoConfig.from_pretrained(cfg_name if cfg_name else "sentence-transformers/paraphrase-TinyBERT-L6-v2")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(
            cfg_name, config=cfg, project_dim=projection_dim, **kwargs
        )

    def forward(
        self, input_ids: T, token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            outputs = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
        else:
            hidden_states = None
            outputs = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

class HFDistilBertEncoder(DistilBertModel):
    def __init__(self, config, project_dim: int = 0):
        DistilBertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs
    ) -> DistilBertModel:
        cfg = DistilBertConfig.from_pretrained(cfg_name if cfg_name else "distilbert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(
            cfg_name, config=cfg, project_dim=projection_dim, **kwargs
        )

    def forward(
        self, input_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.last_hidden_state[:, 0, :]
            hidden_states = outputs.hidden_states
        else:
            hidden_states = None
            outputs = super().forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.last_hidden_state[:, 0, :]

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

class HFRobertaEncoder(RobertaModel):
    def __init__(self, config, project_dim: int = 0):
        RobertaModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs
    ) -> RobertaModel:
        cfg = RobertaConfig.from_pretrained(cfg_name if cfg_name else "roberta-large")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(
            cfg_name, config=cfg, project_dim=projection_dim, **kwargs
        )

    def forward(
        self, input_ids: T, token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size
    
class HFDistilRobertaEncoder(RobertaModel):
    def __init__(self, config, project_dim: int = 0):
        RobertaModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs
    ) -> RobertaModel:
        cfg = RobertaConfig.from_pretrained(cfg_name if cfg_name else "distilroberta-base")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(
            cfg_name, config=cfg, project_dim=projection_dim, **kwargs
        )

    def forward(
        self, input_ids: T, token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class HFElectraEncoder(ElectraModel):
    def __init__(self, config, project_dim: int = 0):
        ElectraModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs
    ) -> ElectraModel:
        print(cfg_name)
        cfg = ElectraConfig.from_pretrained(cfg_name if cfg_name else "google/electra-small-discriminator")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        print(cfg_name)
        model =  cls.from_pretrained(
            cfg_name, config=cfg, project_dim=projection_dim, **kwargs
        )
        print(count_parameters(model))
        return model

    def forward(
        self, input_ids: T, token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = None
            sequence_output = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )[0]

        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states
       # if self.config.output_hidden_states:
       #     sequence_output, pooled_output, hidden_states = super().forward(
       #         input_ids=input_ids,
       #         token_type_ids=token_type_ids,
       #         attention_mask=attention_mask,
       #     )
       # else:
       #     hidden_states = None
       #     sequence_output, pooled_output = super().forward(
       #         input_ids=input_ids,
       #         token_type_ids=token_type_ids,
       #         attention_mask=attention_mask,
       #     )

        #pooled_output = sequence_output[:, 0, :]
        #if self.encode_proj:
        #    pooled_output = self.encode_proj(pooled_output)
        #return sequence_output, pooled_output, hidden_states


    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

class BertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True
    ):
        if isinstance(text, float):
            text = 'nan'
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

class AlbertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: AlbertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True
    ):
        if isinstance(text, float):
            text = 'nan'
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair = text,
                add_special_tokens = add_special_tokens,
                max_length = self.max_length,
                pad_to_max_length = False,
                truncation = True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens = add_special_tokens,
                max_length = self.max_length,
                pad_to_max_length = False,
                truncation = True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

class MobileBertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: MobileBertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True
    ):
        if isinstance(text, float):
            text = 'nan'
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair = text,
                add_special_tokens = add_special_tokens,
                max_length = self.max_length,
                pad_to_max_length = False,
                truncation = True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens = add_special_tokens,
                max_length = self.max_length,
                pad_to_max_length = False,
                truncation = True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

class TinyBertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: AutoTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True
    ):
        if isinstance(text, float):
            text = 'nan'
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair = text,
                add_special_tokens = add_special_tokens,
                max_length = self.max_length,
                pad_to_max_length = False,
                truncation = True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens = add_special_tokens,
                max_length = self.max_length,
                pad_to_max_length = False,
                truncation = True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

class DistilBertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: DistilBertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True
    ):
        if isinstance(text, float):
            text = 'nan'
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair = text,
                add_special_tokens = add_special_tokens,
                max_length = self.max_length,
                pad_to_max_length = False,
                truncation = True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens = add_special_tokens,
                max_length = self.max_length,
                pad_to_max_length = False,
                truncation = True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(
            tokenizer, max_length, pad_to_max=pad_to_max
        )

class ElectraTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(ElectraTensorizer, self).__init__(
            tokenizer, max_length, pad_to_max=pad_to_max
        )