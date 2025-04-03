import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")

tokenizer = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")

text = "!!!!!!!!!!!!!!!!!!!!!!!!!"

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=-1)

predicted_index = torch.argmax(probs, dim=1).item()

predicted_prob = probs[0][predicted_index].item()

labels = model.config.id2label

predicted_label = labels[predicted_index]

for i, prob in enumerate(probs[0]):
    print(f"Class: {labels[i]}, Probability: {prob:.4f}")