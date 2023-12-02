model_without_positional_ids = ['llama', 'llama2', 'vicuna', 'chatglm2', 'chatglm3']
vicuna_system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

lora_target_modules = {
    "chatglm2": ["query_key_value", "dense"],
    "vicuna": ["q_proj","k_proj","v_proj","o_proj","down_proj","gate_proj","up_proj"]
}

freeze_ffn_target_moudles = {
    "chatglm2": ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
    "llama": ["gate_proj", "down_proj", "up_proj"],
    "vicuna": ["gate_proj", "down_proj", "up_proj"]
}

freeze_att_target_moudles = {
    # "chatglm2": ["self_attention.query_key_value", "self_attention.core_attention", "self_attention.query", "self_attention.key_value", "self_attention.dense"],
    "chatglm2": ["self_attention"]
}