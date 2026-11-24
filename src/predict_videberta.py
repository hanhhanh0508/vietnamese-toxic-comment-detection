import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ViDeBERTaToxicPredictor:
    def __init__(
        self,
        ckpt_path: str = "checkpoints/videberta_best.pt",
        threshold_path: str = "configs/videberta_thresholds.json",
        model_name: str = "Fsoft-AIC/videberta-base"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_cols = ["toxic", "hate", "spam", "clean"]

        with open(threshold_path, "r", encoding="utf-8") as f:
            th_dict = json.load(f)
        self.thresholds = [th_dict[col] for col in self.label_cols]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self.label_cols)
        ).to(self.device).float()

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def predict(self, text: str) -> dict:
        clean_text = str(text).replace("_", " ").strip()
        inputs = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        results = {}
        for label, prob, th in zip(self.label_cols, probs, self.thresholds):
            results[label] = {
                "probability": round(prob, 4),
                "threshold": th,
                "predicted": int(prob >= th)
            }
        return results

if __name__ == "__main__":
    predictor = ViDeBERTaToxicPredictor()
    sample = "mày nói chuyện ngu như bò vậy"
    print("Kết quả dự đoán thử:", json.dumps(predictor.predict(sample), ensure_ascii=False, indent=2))