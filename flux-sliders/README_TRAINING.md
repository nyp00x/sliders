# FLUX Slider Training Script

This script converts the FLUX concept sliders training notebook into a runnable Python script with command-line arguments.

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r flux-requirements.txt
```

## Basic Usage

```bash
python train_flux_sliders.py \
    --target_prompt "picture of a person" \
    --positive_prompt "photo of a person, smiling, happy" \
    --negative_prompt "photo of a person, frowning" \
    --slider_name "person-smiling"
```

## Full Example with All Arguments

```bash
python train_flux_sliders.py \
    --model_path "black-forest-labs/FLUX.1-schnell" \
    --target_prompt "picture of a person" \
    --positive_prompt "photo of a person, smiling, happy" \
    --negative_prompt "photo of a person, frowning" \
    --slider_name "person-smiling" \
    --max_train_steps 1000 \
    --learning_rate 0.002 \
    --rank 16 \
    --alpha 1 \
    --eta 2.0 \
    --height 512 \
    --width 512 \
    --output_dir "../../models/fluxsliders" \
    --device "cuda:0"
```

## Arguments

### Required Arguments

- `--target_prompt`: Target/base prompt for the slider (e.g., "picture of a person")
- `--positive_prompt`: Positive direction prompt (e.g., "photo of a person, smiling, happy")
- `--negative_prompt`: Negative direction prompt (e.g., "photo of a person, frowning")
- `--slider_name`: Name for the slider, used in the save path (e.g., "person-smiling")

### Model Arguments

- `--model_path`: Path to pretrained FLUX model (default: "black-forest-labs/FLUX.1-schnell")
  - Options: "black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev", or local path
- `--device`: Device to use for training (default: "cuda:0")

### Training Hyperparameters

- `--max_train_steps`: Number of training steps (default: 1000)
- `--learning_rate`: Learning rate (default: 0.002)
- `--rank`: LoRA rank (default: 16)
- `--alpha`: LoRA alpha (default: 1)
- `--eta`: Training eta parameter for concept guidance (default: 2.0)
- `--batch_size`: Training batch size (default: 1)

### Generation Parameters

- `--height`: Image height (default: 512)
- `--width`: Image width (default: 512)

### Output

- `--output_dir`: Output directory for trained sliders (default: "../../models/fluxsliders")
  - The script will automatically create subdirectories for schnell/dev models

### Advanced Options

- `--train_method`: LoRA training method (default: "xattn")
- `--weighting_scheme`: Timestep weighting scheme (default: "none")
  - Choices: "sigma_sqrt", "logit_normal", "mode", "cosmap", "none"
- `--lr_scheduler`: Learning rate scheduler (default: "constant")
  - Choices: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
- `--lr_warmup_steps`: Number of warmup steps for lr scheduler (default: 200)

## Example Training Commands

### Training a Smile Slider

```bash
python train_flux_sliders.py \
    --target_prompt "picture of a person" \
    --positive_prompt "photo of a person, smiling, happy" \
    --negative_prompt "photo of a person, frowning, sad" \
    --slider_name "smile-slider" \
    --max_train_steps 1000
```

### Training an Age Slider

```bash
python train_flux_sliders.py \
    --target_prompt "portrait of a person" \
    --positive_prompt "portrait of an elderly person, aged, old" \
    --negative_prompt "portrait of a young person, youthful, baby face" \
    --slider_name "age-slider" \
    --max_train_steps 1500 \
    --learning_rate 0.0015
```

### Training a Weather Slider

```bash
python train_flux_sliders.py \
    --target_prompt "landscape photo" \
    --positive_prompt "sunny landscape, bright, clear sky" \
    --negative_prompt "rainy landscape, storm, dark clouds" \
    --slider_name "weather-sunny" \
    --max_train_steps 800
```

### Training with FLUX Dev Model

```bash
python train_flux_sliders.py \
    --model_path "black-forest-labs/FLUX.1-dev" \
    --target_prompt "picture of a room" \
    --positive_prompt "modern minimalist room, clean, organized" \
    --negative_prompt "cluttered messy room, chaotic" \
    --slider_name "room-minimalist" \
    --max_train_steps 1200
```

## Output Structure

After training, the script creates the following structure:

```
output_dir/
├── schnell/  (or dev/)
│   └── flux-{slider_name}/
│       ├── slider_0.pt  (trained LoRA weights)
│       ├── training_loss.png  (loss plot)
│       └── training_config.txt  (training configuration)
```

## Model Types

The script automatically detects the model type:

- **FLUX.1-schnell**: Fast model with 4 inference steps, no guidance
- **FLUX.1-dev**: Higher quality model with 30 inference steps, guidance scale 3.5

## Tips

1. **Prompt Design**: Make sure your positive and negative prompts are clear opposites on a single axis
2. **Training Steps**: Start with 1000 steps and adjust based on results
3. **Learning Rate**: 0.002 works well for most cases, but you can try 0.0015-0.003
4. **Eta Parameter**: Controls the strength of the slider direction. Higher values = stronger effect
5. **Rank**: Higher rank (32, 64) gives more capacity but slower training
6. **Memory**: If you run out of memory, try reducing batch_size or image resolution

## Troubleshooting

### Out of Memory
- Reduce `--height` and `--width` to 256 or 384
- Reduce `--batch_size` to 1
- Reduce `--rank` to 8

### Poor Results
- Increase `--max_train_steps` to 1500-2000
- Adjust `--eta` (try 1.5 or 2.5)
- Make sure prompts are well-contrasted
- Try adjusting `--learning_rate`

### Slow Training
- Use FLUX.1-schnell instead of dev
- Reduce image resolution
- Reduce `--max_train_steps`

## Using the Trained Slider

After training, you can load and use the slider in your inference code:

```python
from utils.lora import LoRANetwork
from utils.custom_flux_pipeline import FluxPipeline

# Load your trained LoRA
network = LoRANetwork(transformer, rank=16, multiplier=1.0, alpha=1)
network.load_state_dict(torch.load("path/to/slider_0.pt"))

# Use in generation with different scales
slider_scale = 2.5  # Positive direction
network.set_lora_slider(scale=slider_scale)

with network:
    image = pipe(
        prompt="your prompt",
        height=512,
        width=512,
        num_inference_steps=4,
        network=network
    )
```

## Citation

If you use this code, please cite the original Concept Sliders paper:

```bibtex
@article{gandikota2023concept,
  title={Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models},
  author={Gandikota, Rohit and others},
  journal={arXiv preprint},
  year={2023}
}
```
