from transformers import (
    LlavaForConditionalGeneration,LlavaConfig,LlavaProcessor,
    CLIPVisionModel,CLIPVisionConfig,CLIPImageProcessor,
    Qwen2ForCausalLM,Qwen2Config,Qwen2Tokenizer
    )
import torch
clip_model_path = "./models/clip-vit-large-patch14-336"
qwen_model_path = "./models/Qwen1.5-1.8B-Chat"
save_dir = "models/llava_clip-L-14-336_Qwen1.5-1.8B"


# 加载视觉模型
vision_model = CLIPVisionModel.from_pretrained(clip_model_path)
vision_config = CLIPVisionConfig.from_pretrained(clip_model_path)
image_processor = CLIPImageProcessor.from_pretrained(clip_model_path)

# 加载语言模型
qwen_model = Qwen2ForCausalLM.from_pretrained(qwen_model_path)
qwen_config = Qwen2Config.from_pretrained(qwen_model_path)
qwen_tokenizer = Qwen2Tokenizer.from_pretrained(qwen_model_path)

# 添加特殊token
qwen_tokenizer.add_special_tokens({"additional_special_tokens":qwen_tokenizer.additional_special_tokens+["<image>"]})

llava_config = LlavaConfig(
    vision_config=vision_config,
    text_config=qwen_config,
    ignore_index=-100,
    image_token_index=151646
)
# 使用从配置初始化 Llava 模型结构
llava_model = LlavaForConditionalGeneration(llava_config)
llava_processor = LlavaProcessor(
    image_processor=image_processor,
    tokenizer=qwen_tokenizer,
    num_additional_image_tokens=1,
    patch_size=vision_config.patch_size,
    vision_feature_select_strategy="default" # 需要指定，否则tokens的维度与image_feature的维度对不上
)

# 复制权重和pad_token_id
llava_model.model.vision_tower = vision_model
llava_model.model.language_model = qwen_model.model
llava_model.config.pad_token_id = qwen_tokenizer.pad_token_id

# 保存模型和配置文件
llava_model.save_pretrained(save_dir)
llava_processor.save_pretrained(save_dir)