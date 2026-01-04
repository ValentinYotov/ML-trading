from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        # Accepts either a string or a list of strings
        if isinstance(news, str):
            texts = [news]
        else:
            texts = news
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        result = torch.nn.functional.softmax(result, dim=-1)
        # For batch, take mean probability and most common sentiment
        avg_probs = result.mean(dim=0)
        probability = avg_probs[torch.argmax(avg_probs)].item()
        sentiment = labels[torch.argmax(avg_probs)]
        return probability, sentiment
    else:
        return 0, labels[-1]


if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(['market didnt care',])
    print(tensor, sentiment)
    print(torch.cuda.is_available())