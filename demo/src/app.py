import gradio as gr
from torchvision import transforms
import onnxruntime as ort
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import pickle
import json
import numpy as np

title = """BioClip2 Quantized Model - Mobile Compatible Demo"""

txt_emb = torch.from_numpy(
    np.load(
        hf_hub_download(
            repo_id="imageomics/TreeOfLife-200M",
            filename="embeddings/txt_emb_species.npy",
            repo_type="dataset",
        )
    )
)
with open(
    hf_hub_download(
        repo_id="imageomics/TreeOfLife-200M",
        filename="embeddings/txt_emb_species.json",
        repo_type="dataset",
    )
) as fd:
    txt_names = json.load(fd)

preprocess_img = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


def format_name(taxon, common):
    if not common:
        return " ".join(taxon)
    else:
        return f"{common}"


with open(
    "/Users/mahan_yt/Documents/work/Dutch-Rose-Media/Projects/AI_test/demo/src/logit_scale.pkl",
    "rb",
) as f:
    logit_scale_value = pickle.load(f)


def infer(image):
    k = 1
    quantized_int8_path = "/Users/mahan_yt/Documents/work/Dutch-Rose-Media/Projects/AI_test/demo/src/bioclip2_model_int8.onnx"

    # Preprocess image
    img_tensor = preprocess_img(image).unsqueeze(0)
    img_np = img_tensor.numpy()

    session = ort.InferenceSession(
        quantized_int8_path, providers=["CPUExecutionProvider"]
    )

    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    img_features_np = session.run([output_name], {input_name: img_np})[0]

    # Convert back to torch for compatibility with existing code
    img_features = torch.from_numpy(img_features_np)
    img_features = F.normalize(img_features, dim=-1)

    # Use the same text embeddings and logit scale from the original model BioClip2
    logits = (logit_scale_value * img_features @ txt_emb).squeeze()
    probs = F.softmax(logits, dim=0)

    topk = probs.topk(k)
    prediction_dict = {
        format_name(*txt_names[i]): prob for i, prob in zip(topk.indices, topk.values)
    }
    return str(*prediction_dict.keys())


demo = gr.Interface(
    title=title,
    fn=infer,
    inputs=gr.Image(label="Input Image", type="pil"),
    outputs=gr.Textbox(label="Predicted Species"),
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="green",
    ),
    description="""Detect and classify animals, plants, fungi, and microorganisms using a mobile-compatible quantized BioClip2 model.""",
)

if __name__ == "__main__":
    demo.launch(mcp_server=True, max_file_size="15mb")
