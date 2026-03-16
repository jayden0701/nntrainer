import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM, set_seed
from PIL import Image
import requests

set_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

model_path = "/home/jayden/Transformers/transformers/src/transformers/models/t5gemma2"

print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

def run_text_to_text(prompt: str, max_new_tokens: int = 50):

    # return_tensors="pt"뜻 : pytorch tensor형태로 리턴
    # return_mm_token_type_ids = true로 설정되어 있어, token_type_ids (tensor)으로 각 token이 이미지인지 단어인지 구분해서 보내줌
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Input tokens: {inputs['input_ids'][0][:20]}")
    print(f"Decoded input: {processor.decode(inputs['input_ids'][0], skip_special_tokens=False)}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            decoder_start_token_id=2
        )

    decoded_full = processor.decode(outputs[0], skip_special_tokens=False)
    print(f"Raw output (full): {decoded_full}")
    return processor.decode(outputs[0], skip_special_tokens=True)

def run_image_text_to_text(image_path_or_url: str, text_prompt: str, max_new_tokens: int = 50):
    try:
        if image_path_or_url.startswith("http"):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            image = Image.open(image_path_or_url)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            decoder_start_token_id=2
        )

    return processor.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    example_text_prompt = "Question: What is the capital of France? Answer in long sentence:"

    print("\n" + "="*50)
    print("Text-to-Text Example")
    print("="*50)
    print(f"Prompt: {example_text_prompt}")
    output = run_text_to_text(example_text_prompt, max_new_tokens=100)
    print(f"Response: {output}")

    example_image_path = "/home/jayden/Transformers/transformers/src/transformers/models/t5gemma2/honey-bee-4.jpg"
    example_image_prompt = "<start_of_image> Describe what you see in this image."

    print("\n" + "="*50)
    print("Image+Text-to-Text Example")
    print("="*50)
    print(f"Image path: {example_image_path}")
    print(f"Prompt: {example_image_prompt}")
    output = run_image_text_to_text(example_image_path, example_image_prompt, max_new_tokens=100)
    print(f"Response: {output}")
    print("="*50)
