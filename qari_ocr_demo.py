from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info


def load_model(model_name="NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def extract_text_from_image(model, processor, image_path, prompt=None, max_tokens=2000):
    if prompt is None:
        prompt = (
            "Below is the image of one page of a document. Just return the plain text "
            "representation of this document as if you were reading it naturally. Do not hallucinate."
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    model.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return output_text


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qari_ocr_demo.py <image_file>")
    else:
        model, processor = load_model()
        result = extract_text_from_image(model, processor, sys.argv[1])
        print(result)
