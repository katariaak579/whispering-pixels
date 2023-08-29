import gradio as gr
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration, VisionEncoderDecoderModel, InstructBlipForConditionalGeneration
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"

git_processor_large_coco = AutoProcessor.from_pretrained("microsoft/git-large-coco")
git_model_large_coco = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").to(device)

# blip_processor_large = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
# blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b-coco", device_map="auto", load_in_4bit=True, torch_dtype=torch.float16)

# instructblip_processor = AutoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
# instructblip_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map="auto", load_in_4bit=True, torch_dtype=torch.float16)

def generate_caption(processor, model, image, tokenizer=None, use_float_16=False):
    inputs = processor(images=image, return_tensors="pt").to(device)

    if use_float_16:
        inputs = inputs.to(torch.float16)
    
    generated_ids = model.generate(pixel_values=inputs.pixel_values, num_beams=3, max_length=20, min_length=5) 

    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
   
    return generated_caption


def generate_caption_blip2(processor, model, image, replace_token=False):    
    prompt = "A photo of"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=model.device, dtype=torch.float16)

    generated_ids = model.generate(**inputs,
                                   num_beams=5, max_length=50, min_length=1, top_p=0.9,
                                   repetition_penalty=1.5, length_penalty=1.0, temperature=1)
    if replace_token:
        generated_ids[generated_ids == 0] = 2 
    
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def generate_captions(image):
    image = Image.open(image)
    caption_git_large_coco = generate_caption(git_processor_large_coco, git_model_large_coco, image)

    # caption_blip_large = generate_caption(blip_processor_large, blip_model_large, image)

    # caption_blip2 = generate_caption_blip2(blip2_processor, blip2_model, image).strip()

    # caption_instructblip = generate_caption_blip2(instructblip_processor, instructblip_model, image, replace_token=True)

    return  caption_git_large_coco

   
