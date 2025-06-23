from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

def run_inference(examples, new_story):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a highly skilled QA professional..."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Below are one or more user stories..."},
                {"type": "text", "text": examples},
                {"type": "text", "text": "Now generate positive and negative test cases..."},
                {"type": "text", "text": new_story},
                {"type": "text", "text": (
                    "Generate them as list of objects with id, title, priority, category..."
                )}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded
