"""
Persian G2P (Grapheme-to-Phoneme) Inference Module
Powered by ByT5 fine-tuned on KaamelDict.

Hugging Face: https://huggingface.co/AminMadani/persian-byt5-g2p
"""

import argparse
import sys
import torch
from typing import List, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class PersianG2P:
    """
    Persian Grapheme-to-Phoneme converter using fine-tuned ByT5.
    """
    def __init__(self, model_id: str = "AminMadani/persian-byt5-g2p", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Persian G2P model from '{model_id}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def predict(self, text: Union[str, List[str]], max_length: int = 128, num_beams: int = 2) -> Union[str, List[str]]:
        """
        Convert Persian text (word or list of words) to space-separated phonemes.
        """
        is_single = isinstance(text, str)
        inputs_list = [text] if is_single else text

        # ByT5 byte-level tokenization
        encoded = self.tokenizer(
            inputs_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=64
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True if num_beams > 1 else False
            )

        decoded = [self.tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
        return decoded[0] if is_single else decoded


def main():
    parser = argparse.ArgumentParser(description="Persian G2P using fine-tuned ByT5")
    parser.add_argument("--text", "-t", type=str, help="Single word or phrase to transcribe")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive CLI session")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--beams", type=int, default=2, help="Number of beams for beam search")
    args = parser.parse_args()

    g2p = PersianG2P(device=args.device)

    if args.text:
        phonemes = g2p.predict(args.text, num_beams=args.beams)
        print(f"\n{args.text} -> {phonemes}")
        return

    if args.interactive or not sys.stdin.isatty():
        print("\nInteractive Persian G2P Shell. Type a Persian word (or 'exit' to quit):")
        while True:
            try:
                line = input("\n> ").strip()
                if not line or line.lower() in ["exit", "quit", "q"]:
                    break
                print(f"{line} -> {g2p.predict(line, num_beams=args.beams)}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting.")
    else:
        # Default showcase
        demo_words = ["دانشگاه", "کامپیوتر", "خوش‌آمدید", "هوش مصنوعی", "تلفظ", "واترپولو", "دربازکن"]
        print("\nRunning demonstration on sample Persian words:")
        for w in demo_words:
            print(f"{w:15} -> {g2p.predict(w, num_beams=args.beams)}")


if __name__ == "__main__":
    main()
