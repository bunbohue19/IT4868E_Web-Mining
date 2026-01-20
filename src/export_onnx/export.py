from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import onnx

# Model and output paths
model_name = "D:/Lafin/Personal/Ths/Web Mining/IT4868E_Web-Mining/model/fine_tuned_infoNCE_gte_multilingual_japanese"
output_dir = "D:/Lafin/Personal/Ths/Web Mining/IT4868E_Web-Mining/model/infoNCE_gte_multilingual_japanese"
output_dir = Path(output_dir)
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

    # Export using torch.onnx.export
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
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid!")

    # Test inference with ONNX Runtime
    print("\nTesting ONNX model inference...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    test_text = "これはテストです。This is a test."
    test_inputs = tokenizer(test_text, return_tensors="np", padding=True, truncation=True)

    ort_inputs = {
        "input_ids": test_inputs["input_ids"],
        "attention_mask": test_inputs["attention_mask"]
    }
    outputs = session.run(None, ort_inputs)
    print(f"✓ Inference successful! Output shape: {outputs[0].shape}")

except Exception as e:
    print(f"\n✗ Error during export: {e}")
    import traceback
    traceback.print_exc()
