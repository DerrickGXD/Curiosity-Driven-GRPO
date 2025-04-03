from transformers import AutoTokenizer, AutoModel
import torch
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', device=device)
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)

from flask import Flask, request, jsonify
from urllib.parse import unquote

app = Flask(__name__)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@app.route("/data", methods=["GET"])
def get_data():
    text = request.args.get("text")
    text = unquote(text)
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return jsonify({"message": sentence_embeddings.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)