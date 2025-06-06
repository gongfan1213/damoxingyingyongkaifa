from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='glm-4-9b-hf-chat',
        path='/data/whwang22/pretrained_model/glm-4-9b-chat',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
    )
]