import os
import unicodedata
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, origins="*")

MODEL_FOLDER = "./models/English-Hausa_NLLB_FT_model"
BASE_MODEL = "facebook/nllb-200-distilled-600M"
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "hau_Latn"
DEFAULT_PORT = 5003

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)

if device == "cuda":
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
else:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    )

model = PeftModel.from_pretrained(base_model, MODEL_FOLDER)
model.eval()

if device == "cpu":
    model.to("cpu")

print(f"Model ready! Device: {device}")


def translate(text, source_lang=SOURCE_LANG, target_lang=TARGET_LANG):
    tokenizer.src_lang = source_lang
    inputs = tokenizer(
        text, return_tensors="pt",
        padding=True, truncation=True, max_length=512
    )
    if device == "cpu":
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            max_length=512
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return unicodedata.normalize('NFC', decoded)


@app.route("/")
def index():
    return jsonify({
        "model": "drakensberg85/English-Hausa_NLLB_FT_model",
        "org": "LinguAfrika",
        "capabilities": ["translation"],
        "endpoints": ["/", "/health", "/translate", "/translate/batch", "/languages", "/help"]
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "English-Hausa_NLLB_FT_model", "device": device})


@app.route("/translate", methods=["POST", "OPTIONS"])
def translate_endpoint():
    if request.method == "OPTIONS":
        r = jsonify({})
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        r.headers["Access-Control-Allow-Methods"] = "POST"
        return r
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    try:
        result = translate(
            data["text"],
            data.get("source_lang", SOURCE_LANG),
            data.get("target_lang", TARGET_LANG)
        )
        resp = jsonify({"translation": result})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/translate/batch", methods=["POST", "OPTIONS"])
def translate_batch():
    if request.method == "OPTIONS":
        r = jsonify({})
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        r.headers["Access-Control-Allow-Methods"] = "POST"
        return r
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400
    try:
        results = [translate(t, data.get("source_lang", SOURCE_LANG), data.get("target_lang", TARGET_LANG)) for t in data["texts"]]
        resp = jsonify({"translations": results})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/languages")
def languages():
    return jsonify({
        "source": SOURCE_LANG,
        "target": TARGET_LANG,
        "supported": ["eng_Latn", "hau_Latn"]
    })


@app.route("/help")
def help_endpoint():
    return jsonify({
        "endpoints": {
            "GET /health": "Check server status",
            "POST /translate": {"body": {"text": "string", "source_lang": "optional", "target_lang": "optional"}},
            "POST /translate/batch": {"body": {"texts": ["array of strings"]}},
            "GET /languages": "List supported language codes"
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
