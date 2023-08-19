def init_hf_bert_biencoder(args, **kwargs):
    from .hf_models import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)

def init_hf_albert_biencoder(args, **kwargs):
    from .hf_models import get_albert_biencoder_components
    return get_albert_biencoder_components(args, **kwargs)

def init_hf_mobilebert_biencoder(args, **kwargs):
    from .hf_models import get_mobilebert_biencoder_components
    return get_mobilebert_biencoder_components(args, **kwargs)

def init_hf_roberta_biencoder(args, **kwargs):
    from .hf_models import get_roberta_biencoder_components
    return get_roberta_biencoder_components(args, **kwargs)

def init_hf_electra_biencoder(args, **kwargs):
    from .hf_models import get_electra_biencoder_components
    return get_electra_biencoder_components(args, **kwargs)


def init_hf_bert_tenzorizer(args, **kwargs):
    from .hf_models import get_bert_tensorizer
    return get_bert_tensorizer(args)

def init_hf_albert_tenzorizer(args, **kwargs):
    from .hf_models import get_albert_tensorizer
    return get_albert_tensorizer(args)

def init_hf_mobilebert_tenzorizer(args, **kwargs):
    from .hf_models import get_mobilebert_tensorizer
    return get_mobilebert_tensorizer(args)

def init_hf_electra_tenzorizer(args, **kwargs):
    from .hf_models import get_electra_tensorizer
    return get_electra_tensorizer(args)

def init_hf_roberta_tenzorizer(args, **kwargs):
    from .hf_models import get_roberta_tensorizer
    return get_roberta_tensorizer(args)

BIENCODER_INITIALIZERS = {
    'hf_bert': init_hf_bert_biencoder,
    'hf_electra': init_hf_electra_biencoder,
    'hf_roberta': init_hf_roberta_biencoder,
    'hf_albert': init_hf_albert_biencoder,
    'hf_mobilebert': init_hf_mobilebert_biencoder
}

TENSORIZER_INITIALIZERS = {
    'hf_bert': init_hf_bert_tenzorizer,
    'hf_electra': init_hf_electra_tenzorizer,
    'hf_roberta': init_hf_roberta_tenzorizer,
    'pytext_bert': init_hf_bert_tenzorizer,  # using HF's code as of now
    'fairseq_roberta': init_hf_roberta_tenzorizer,  # using HF's code as of now
    'hf_albert': init_hf_albert_tenzorizer,
    'hf_mobilebert': init_hf_mobilebert_tenzorizer
}

def init_comp(initializers_dict, type, args, **kwargs):
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError('unsupported model type: {}'.format(type))

def init_biencoder_components(encoder_type: str, args, **kwargs):
    return init_comp(BIENCODER_INITIALIZERS, encoder_type, args, **kwargs)

def init_tenzorizer(encoder_type: str, args, **kwargs):
    return init_comp(TENSORIZER_INITIALIZERS, encoder_type, args, **kwargs)