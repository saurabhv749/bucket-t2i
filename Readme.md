# Multi-Resolution Text-to-Image Diffusion Model Trainer

A PyTorch-based training framework for fine-tuning Stable Diffusion models on custom datasets using **aspect ratio bucketing**. This approach efficiently handles variable image dimensions while minimizing computational waste and memory overhead.

## 🎯 Features

- **Aspect Ratio Bucketing**: Intelligently groups images by aspect ratio into resolution buckets, ensuring each batch contains images of the same dimensions for zero padding waste
- **Multi-Resolution Support**: Seamlessly handles square, portrait, and landscape images with near-identical pixel budgets
- **Memory Efficient**: Reduces VRAM usage by maintaining uniform batch dimensions and supporting gradient checkpointing
- **HuggingFace Hub Integration**: Load datasets directly from the HuggingFace Hub or use local datasets
- **Single & Multi-GPU Training**: Full support for distributed training with `accelerate`
- **Advanced Training Features**:
  - Mixed precision training (fp16/bf16)
  - EMA (Exponential Moving Average) model support
  - SNR-weighted loss scaling
  - DREAM training method
  - Learning rate scheduling with warmup
  - Checkpoint saving and resuming
- **Validation & Logging**: Built-in validation during training with TensorBoard and Weights & Biases integration

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- See `requirements.txt` for detailed dependencies

## 🚀 Installation

```bash
# Clone the Repository
git clone https://github.com/saurabhv749/bucket-t2i.git
cd bucket-t2i
#Install Dependencies
pip install -r requirements.txt
```

## 📚 Dataset

Your dataset should have 'image' and 'text' columns
Example: `sam749/midjourney-civitai-256`


## ⚙️ Configuration Options

### Dataset Configuration
- `--dataset_name`: HuggingFace dataset identifier or local path
- `--train_data_dir`: Path to local dataset folder
- `--image_column`: Column name containing images (default: "image")
- `--caption_column`: Column name containing captions (default: "text")

### Bucketing Configuration
- `--min_px`: Minimum image dimension in pixels (default: 512)
- `--max_px`: Maximum image dimension in pixels (default: 1024)
- `--max_px_area`: Maximum pixel area for buckets (default: 768×768)

### Training Configuration
- `--train_batch_size`: Batch size per device (default: 16)
- `--num_train_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--mixed_precision`: Precision mode - "no", "fp16", "bf16" (default: None)

### Validation Configuration
- `--validation_epochs`: Run validation every N epochs
- `--validation_prompts`: Prompts for validation (space-separated)
- `--val_image_width`: Validation image width (default: 192)
- `--val_image_height`: Validation image height (default: 320)


## 🏗️ Architecture Overview

### Core Components

#### 1. **Bucket DataLoader** (`bucket_dataloader.py`)
- **Bucket Generation**: Creates resolution pairs (W, H) with balanced pixel budgets
- **Aspect Ratio Assignment**: Intelligently assigns images to appropriate buckets
- **T2I Dataset**: Loads and processes image-caption pairs
- **Aspect Ratio Bucket Sampler**: Groups images by bucket for efficient batching
- **Collate Function**: Stacks same-bucket samples without padding

#### 2. **Training Script** (`train_t2i.py`)
- Model loading and preparation
- Training loop with distributed support
- Validation with generated images
- Checkpoint management
- Hub integration for model sharing

### Key Concepts

**Aspect Ratio Bucketing**: Instead of resizing all images to a fixed resolution, the framework groups images by aspect ratio into buckets with similar pixel budgets. This:\n- Eliminates padding waste (more efficient GPU memory usage)\n- Preserves image details better than uniform resizing\n- Creates uniform batch dimensions automatically

## 💡 Usage Examples

### Example 1: Fine-tune on Custom Dataset
```bash
accelerate launch train_t2i.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
 --train_data_dir="./my_dataset" --output_dir="./my_model" \
 --train_batch_size=8 --num_train_epochs=50 \
 --validation_prompts "my custom prompt 1" "my custom prompt 2"
```


## 📝 Citation

This project implements aspect ratio bucketing for efficient diffusion model training. The technique helps reduce computational waste and improves training efficiency.
The training script improves upon `/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.