from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MBartForConditionalGeneration, MBartTokenizerFast
import torch
import os, uvicorn

# Define the input schema
class TranslationRequest(BaseModel):
    src: str
    language: str

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer
model_path = "feedlight42/mbart25-text2picto"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBartTokenizerFast.from_pretrained(model_path)
model = model.to(device)

@app.post("/translate")
def translate(request: TranslationRequest):
    """
    Translate text to target language and generate pictogram tokens.
    """
    inputs = tokenizer(request.src, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate translation
    translated_tokens = model.generate(**inputs)
    tgt_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    # Dummy pictogram mapping (replace with real logic)
    pictograms = [123, 456]  # Example pictogram IDs for `tgt_sentence`

    return {
        "src": request.src,
        "language": request.language,
        "tgt": tgt_sentence,
        "pictograms": pictograms
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if no PORT is set
    uvicorn.run(app, host="0.0.0.0", port=port)
