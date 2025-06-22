import os
import sys
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM


def get_comfyui_root():
    main_module = sys.modules.get('__main__')
    if main_module and hasattr(main_module, '__file__'):
        main_path = os.path.abspath(main_module.__file__)
        root_dir = os.path.dirname(main_path)
        return root_dir
    return None

def tensor2pil(image):
    """Convert a ComfyUI tensor to PIL Image"""
    if image.dim() == 4:
        image = image.squeeze(0)
    image = image.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def pil2tensor(image):
    """Convert PIL Image to ComfyUI tensor"""
    array = np.array(image).astype(np.float32) / 255.0
    if array.ndim == 3:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array)

# Register model_path
models_dir = os.path.join(get_comfyui_root(), "models")
ovis_dir = os.path.join(models_dir, "checkpoints", "Ovis2")
os.makedirs(ovis_dir, exist_ok=True)

# Add Ovis models folder to folder_paths
# if "ovis" not in folder_paths.folder_names_and_paths:
#     folder_paths.folder_names_and_paths["ovis"] = ([ovis_dir], folder_paths.supported_pt_extensions)

class Ovis2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["AIDC-AI/Ovis2-34B", "AIDC-AI/Ovis2-16B", "AIDC-AI/Ovis2-8B", "AIDC-AI/Ovis2-4B",
                               "AIDC-AI/Ovis2-2B", "AIDC-AI/Ovis2-1B"], {"default": "AIDC-AI/Ovis2-4B"}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "max_token_length": ("INT", {"default": 32768, "min": 2048, "max": 65536}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": (["enable", "disable"], {"default": "enable"}),
            }
        }
    
    RETURN_TYPES = ("OVIS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Ovis2"

    def download_model(self, model_name):
        """Download the model files from Hugging Face if they don't exist locally."""
        local_dir = os.path.join(ovis_dir, model_name.split('/')[-1])
        
        print(f"Downloading Ovis2 model: {model_name} to {local_dir}")
        try:
            # Create a complete snapshot of the repository locally
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False  # Use actual files instead of symlinks for better compatibility
            )
            print(f"Successfully downloaded {model_name} to {local_dir}")
            return local_dir
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise RuntimeError(f"Failed to download model {model_name}. Error: {str(e)}")

    def check_model_files(self, model_name):
        """Check if the model files already exist locally."""
        local_dir = os.path.join(ovis_dir, model_name.split('/')[-1])
        
        # Check for config.json as a basic indicator that the model exists
        config_path = os.path.join(local_dir, "config.json")
        return os.path.exists(config_path), local_dir

    def load_model(self, model_name, precision, max_token_length, device, auto_download):
        print(f"Loading Ovis2 model: {model_name}")
        
        # Set precision
        if precision == "bfloat16":
            dtype = torch.bfloat16
        elif precision == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Check if model exists locally
        model_exists, local_dir = self.check_model_files(model_name)
        
        # Download model if it doesn't exist and auto_download is enabled
        if not model_exists and auto_download == "enable":
            self.download_model(model_name)

        # Load the model
        try:
            # First try loading from local directory
            if model_exists or auto_download == "enable":
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        local_dir,
                        torch_dtype=dtype,
                        multimodal_max_length=max_token_length,
                        trust_remote_code=True
                    ).to(device)
                except Exception as e:
                    print(f"Error loading from local directory, falling back to HuggingFace: {str(e)}")
                    # Fall back to loading directly from HuggingFace
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        multimodal_max_length=max_token_length,
                        trust_remote_code=True
                    ).to(device)
            else:
                # Load directly from HuggingFace
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    multimodal_max_length=max_token_length,
                    trust_remote_code=True
                ).to(device)
            
            # Get tokenizers
            text_tokenizer = model.get_text_tokenizer()
            visual_tokenizer = model.get_visual_tokenizer()
            
            return ({"model": model, "text_tokenizer": text_tokenizer, "visual_tokenizer": visual_tokenizer},)
        except Exception as e:
            print(f"Error loading Ovis2 model: {str(e)}")
            raise e


class Ovis2ImageCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS2_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.01}),
                "do_sample": (["true", "false"], {"default": "true"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "generate_caption"
    CATEGORY = "Ovis2"

    def generate_caption(self, model, image, prompt, max_new_tokens, temperature, top_p, do_sample):
        model_data = model
        model = model_data["model"]
        text_tokenizer = model_data["text_tokenizer"]
        visual_tokenizer = model_data["visual_tokenizer"]

        # 使用 tensor2pil 进行图像转换
        if image.dim() == 4:
            pil_image = tensor2pil(image[0])
        else:
            pil_image = tensor2pil(image)

        # 准备查询
        query = f"<image>\n{prompt}"

        try:
            # 预处理输入（按照原始代码方式）
            _, input_ids, pixel_values = model.preprocess_inputs(
                query,
                [pil_image],
                max_partition=9
            )

            # 创建注意力掩码
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)

            # 准备模型输入
            input_ids = input_ids.unsqueeze(0).to(model.device)
            attention_mask = attention_mask.unsqueeze(0).to(model.device)

            if pixel_values is not None:
                pixel_values = pixel_values.to(
                    dtype=visual_tokenizer.dtype,
                    device=model.device
                )
            pixel_values = [pixel_values]

            # 转换布尔值参数
            do_sample_bool = do_sample.lower() == "true"

            # 生成输出
            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample_bool,
                    "top_p": top_p,
                    "temperature": temperature,
                    "eos_token_id": model.generation_config.eos_token_id,
                    "pad_token_id": text_tokenizer.pad_token_id,
                    "use_cache": True
                }

                # 调用generate方法（注意参数顺序）
                output_ids = model.generate(
                    input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )[0]

                # 解码生成的文本
                generated_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)

                # 提取生成的响应（移除查询部分）
                # response = generated_text[len(query):].strip()

                return (generated_text.strip(),)

        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            import traceback
            traceback.print_exc()
            return (f"Error generating caption: {str(e)}",)

class Ovis2MultiImageInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS2_MODEL",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe each image.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "max_descript": ("INT", {"default": 3, "min": 3, "max": 240, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "process_multi_images"
    CATEGORY = "Ovis2"

    def process_multi_images(self, model, images, prompt, max_new_tokens, temperature, max_descript):
        model_data = model
        model_obj = model_data["model"]
        text_tokenizer = model_data["text_tokenizer"]
        visual_tokenizer = model_data["visual_tokenizer"]

        # 1. Convert ComfyUI image frames to PIL images
        pil_images = []

        # 2. Apply frame skipping and max frames limit
        total_frames = images.size(0)

        # 3. Convert selected frames to PIL
        for idx in range(total_frames):
            if idx >= max_descript:
                break
            frame_tensor = images[idx]
            pil_img = tensor2pil(frame_tensor)
            pil_images.append(pil_img)

        # 4. Prepare query with image placeholders
        max_partition_in = 5
        num_processed = len(pil_images)
        text = 'Describe each image.'
        query = '\n'.join([f'Image {i+1}: <image>' for i in range(num_processed)]) + '\n' + text

        try:
            # 5. Preprocess inputs
            prompt_text, input_ids, pixel_values = model_obj.preprocess_inputs(
                query,
                pil_images,
                max_partition=max_partition_in
            )

            # 6. Prepare attention mask
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model_obj.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model_obj.device)

            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=model_obj.device)
            pixel_values = [pixel_values]  # Wrap in list as expected by model

            # 7. Configure generation parameters
            do_sample = temperature > 0.1  # Enable sampling only when temperature is meaningful
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature if do_sample else None,
                "top_p": 0.9 if do_sample else None,
                "top_k": 50 if do_sample else None,
                "repetition_penalty": 1.1,
                "eos_token_id": model_obj.generation_config.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True
            }

            # 8. Generate description
            with torch.inference_mode():
                output_ids = model_obj.generate(
                    input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )[0]

                # Decode and clean output
                output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                # Remove the input prompt from the output
                # generated_text = output[len(prompt_text):].strip()

            return (output.strip(),)
        except Exception as e:
            return (f"Error: {str(e)}",)


class Ovis2VideoFramesDescription:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS2_MODEL",),
                "frames": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe what's happening in this video.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "frame_skip": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),
                "max_frames": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "descript_video"
    CATEGORY = "Ovis2"

    def descript_video(self, model, frames, prompt, max_new_tokens, temperature, frame_skip, max_frames):
        model_data = model
        model_obj = model_data["model"]
        text_tokenizer = model_data["text_tokenizer"]
        visual_tokenizer = model_data["visual_tokenizer"]

        # 1. Convert ComfyUI image frames to PIL images
        pil_images = []
        total_frames = frames.size(0)

        # 2. Apply frame skipping and max frames limit
        selected_indices = []

        if frame_skip == 0:
            for idx in range(total_frames):
                if len(selected_indices) >= max_frames:
                    break
                selected_indices.append(idx)
        else:
            for idx in range(0, total_frames, frame_skip):
                if len(selected_indices) >= max_frames:
                    break
                selected_indices.append(idx)

        # 3. Convert selected frames to PIL
        for idx in selected_indices:
            frame_tensor = frames[idx]
            pil_img = tensor2pil(frame_tensor)
            pil_images.append(pil_img)

        # 4. Prepare query with image placeholders
        image_placeholders = "\n".join(["<image>"] * len(pil_images))
        query = f"{image_placeholders}\n{prompt}"

        try:
            # 5. Preprocess inputs
            prompt_text, input_ids, pixel_values = model_obj.preprocess_inputs(
                query,
                pil_images,
                max_partition=1  # Use partitioning for video frames
            )

            # 6. Prepare attention mask
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model_obj.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model_obj.device)

            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=model_obj.device)
            pixel_values = [pixel_values]  # Wrap in list as expected by model

            # 7. Configure generation parameters
            do_sample = temperature > 0.1  # Enable sampling only when temperature is meaningful
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature if do_sample else None,
                "top_p": 0.9 if do_sample else None,
                "top_k": 50 if do_sample else None,
                "repetition_penalty": 1.1,
                "eos_token_id": model_obj.generation_config.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True
            }

            # 8. Generate description
            with torch.inference_mode():
                output_ids = model_obj.generate(
                    input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )[0]

                # Decode and clean output
                output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                # Remove the input prompt from the output
                # generated_text = output[len(prompt_text):].strip()

            return (output.strip(),)
        except Exception as e:
            return (f"Error: {str(e)}",)



# Register the nodes
NODE_CLASS_MAPPINGS = {
    "Ovis2ModelLoader": Ovis2ModelLoader,
    "Ovis2ImageCaption": Ovis2ImageCaption,
    "Ovis2MultiImageInput": Ovis2MultiImageInput,
    "Ovis2VideoFramesDescription": Ovis2VideoFramesDescription,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ovis2ModelLoader": "Load Ovis2 Model",
    "Ovis2ImageCaption": "Ovis2 Image Caption",
    "Ovis2MultiImageInput": "Ovis2 Multi-Image Analysis",
    "Ovis2VideoFramesDescription": "Ovis2 Video Frames Description",
}
