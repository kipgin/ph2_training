import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images for fine-tuning the model.
    It "lazy loads" the images from the given directories.
    
    Attributes:
        instance_data_root (str): Path to instance images directory.
        instance_prompt (str): Prompt description of the instance images.
        tokenizer (CLIPTokenizer): Tokenizer for tokenizing the prompt.
        class_data_root (str, optional): Path to class images directory.
        class_prompt (str, optional): Prompt description of the class images.
        size (int): Size to resize the images to.
        center_crop (bool): Whether to use center cropping or random cropping.
    """
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=True,
    ):
        self.instance_data_root = instance_data_root
        if not os.path.exists(self.instance_data_root):
            raise ValueError(f"Instance data root directory does not exist: {self.instance_data_root}")

        # Gather instance images path
        self.instance_images_path = [
            os.path.join(self.instance_data_root, f) for f in os.listdir(self.instance_data_root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        self.num_instance_images = len(self.instance_images_path)
        if self.num_instance_images == 0:
            raise ValueError(f"No valid images found in instance data root: {self.instance_data_root}")

        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self._length = self.num_instance_images

        # Handle optional class data root for prior preservation
        self.class_data_root = class_data_root
        if self.class_data_root is not None:
            if not os.path.exists(self.class_data_root):
                raise ValueError(f"Class data root directory does not exist: {self.class_data_root}")
            self.class_images_path = [
                os.path.join(self.class_data_root, f) for f in os.listdir(self.class_data_root)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
            ]
            self.num_class_images = len(self.class_images_path)
            # Length is the maximum of instance and class images
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_images_path = None
            self.num_class_images = 0

        # Preprocessing transforms (Standard for Stable Diffusion, normalizes to [-1, 1])
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        # Load and preprocess instance image
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        
        # Tokenize instance prompt
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        # Load and preprocess class image if available
        if self.class_images_path is not None and self.num_class_images > 0:
            class_image_path = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            
            # Tokenize class prompt
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    """
    A simple dataset to generate a list of prompts for generating class/prior images.
    """
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class DreamBoothCollator:
    """
    Data collator class to assemble batches from the DreamBoothDataset.
    It stacks pixel values and pads text tokens dynamically.
    """
    def __init__(self, tokenizer, with_prior_preservation=False):
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation

    def __call__(self, examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Stack and pad class images if prior preservation is active
        if self.with_prior_preservation and "class_images" in examples[0]:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # Dynamic padding of token sequences in the batch
        padded_tokens = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        batch = {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
        }

        if "attention_mask" in padded_tokens:
            batch["attention_mask"] = padded_tokens.attention_mask

        return batch
