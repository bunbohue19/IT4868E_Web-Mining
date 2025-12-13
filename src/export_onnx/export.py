import os
from pathlib import Path
from optimum.exporters.onnx import main_export
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch

# Model and output paths
model_name = "jaeyong2/gte-multilingual-base-Ja-embedding"
output_dir = Path(os.environ.get("HF_HOME")) / "gte-multilingual-base-JA_onnx"
output_dir = output_dir.expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading model from: {model_name}")
print(f"Output directory: {output_dir}")

# Load the model and tokenizer with trust_remote_code
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input for tracing
    dummy_text = "This is a sample text for model export"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
    
    print("\nExporting to ONNX...")
    
    # Export using torch.onnx.export directly
    onnx_path = output_dir / "model.onnx"
    
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    # Save tokenizer and config
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    
    print(f"\n✓ Model successfully exported to: {onnx_path}")
    print(f"✓ Tokenizer and config saved to: {output_dir}")
    
    # Verify the export
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid!")
    
except Exception as e:
    print(f"\n✗ Error during export: {e}")
    import traceback
    traceback.print_exc()