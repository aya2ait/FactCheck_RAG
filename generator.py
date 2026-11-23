
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import config

_SYSTEM = (
    "You are a factual question-answering assistant. "
    "Answer using ONLY the passages provided. "
    "After each factual claim add a citation tag [1], [2], ... "
    "referring to the passage number. "
    "If passages lack information, say so explicitly."
)

def _build_prompt(question, passages):
    block = "\n\n".join(
        f"[{i+1}] (Title: {p.get('title','')}) {p['text']}"
        for i, p in enumerate(passages)
    )
    return (
        f"Passages:\n{block}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with inline citations):"
    )

class Generator:
    def __init__(self):
        print(f"Loading {config.GEN_MODEL} ...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(config.GEN_MODEL)
        self._model = AutoModelForCausalLM.from_pretrained(
            config.GEN_MODEL,
            quantization_config=bnb,
            device_map="auto",
        )
        print("Generator ready.")

    def generate(self, question: str, passages: list) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _build_prompt(question, passages)},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=False,
            )
        decoded = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return decoded.strip()
