from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor
import torch
from PIL import Image


llava_mode_path = "models/llava_clip-L-14-336_Qwen1.5-1.8B"
llava_processor = LlavaProcessor.from_pretrained(llava_mode_path, use_fast=True)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    llava_mode_path, device_map="cpu"
)


prompt_text = "<image>\nWhat are these?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]
prompt = llava_processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)


image_path = "test.jpg"
image = Image.open(image_path)


inputs = llava_processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = llava_model.generate(**inputs, max_new_tokens=15)
llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
for tk in inputs.keys():
    inputs[tk] = inputs[tk].to(llava_model.device)
generate_ids = llava_model.generate(**inputs, max_new_tokens=20)
gen_text = llava_processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

print(gen_text)
