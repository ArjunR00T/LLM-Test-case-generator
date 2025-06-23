

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading from cache")
faiss_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
)
print("Model loaded successfully")

with open("/content/sample_data.json") as f:
    data = json.load(f)

corpus = [entry["user_story"] for entry in data]

corpus_embeddings = faiss_model.encode(corpus, normalize_embeddings=True)

index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

def get_similar_examples(query, k=3):
    print("Searching for similar examples...")
    query_embedding = faiss_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, k)
    results = [data[i] for i in indices[0]]
    return results

def gemma_model(examples, new_story):
    print(examples)
    print(new_story)
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a highly skilled QA professional with expertise in writing precise, relevant, and edge-covering test cases."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Below are one or more user stories along with their related test cases. Study them carefully:"},
                {"type": "text", "text": examples},
                {"type": "text", "text": "Now, refer to the structure, pattern, and logic of the above test cases and generate positive and negative test cases for the following new user story:"},
                {"type": "text", "text": new_story},
                {
                    "type": "text",
                    "text": (
                        "You can take inspiration from the existing test cases or use your own judgment to generate high-quality, meaningful test cases. "
                        "Ensure all generated test cases are relevant to the user story. Do not include test steps or explanations — just list the test cases as bullet points or in a list."
                    )
                },
                {
                    "type":"text", "text":("Generate them as list of objects which i can pass to my frontend with each object containing an id,title,priority(Critical,High,Medium,Low) and category(eg: Functional,Validation,Security,Error-handling..etc)"
                    "And only give it as output, nothing else, so that i can directly pass it to my response body from my backend to frontend")
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

import re
import json

def clean_and_parse_partial_json(raw_output):
    print(raw_output)

    if isinstance(raw_output, tuple):
        print("⚠️ Warning: raw_output is a tuple, extracting first element")
        raw_output = raw_output[0]  # FIX: use first item
        
    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]  # Fixing your tuple issue

    # 1. Remove markdown code block indicators
    cleaned = re.sub(r"```json|```", "", raw_output.strip())

    # 2. Try to extract the JSON array from anywhere in the string
    match = re.search(r"\[\s*{.*", cleaned, re.DOTALL)
    if not match:
        print("❌ No JSON array detected")
        return None

    json_str = match.group()

    # 3. Try parsing normally
    try:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(json_str)
        return parsed
    except json.JSONDecodeError:
        pass

    # 4. Try trimming the trailing part (most common fix)
    try:
        # Remove trailing comma and incomplete object
        json_str = re.sub(r",\s*{[^}]*$", "", json_str.strip())
        json_str += "]"  # close the array
        parsed = json.loads(json_str)
        return parsed
    except Exception as e:
        print("❌ Final parse failed:", e)
        return None

# !wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
# !chmod +x cloudflared

# In-memory fake redis
task_results = {} 

def generate_tc(uuid,new_story):
    print("Model execution started..")
    task_results[uuid] = {"status": "processing", "result": None}
    similar_examples = get_similar_examples(new_story)
    generated_test_cases = clean_and_parse_partial_json(gemma_model(json.dumps(similar_examples), new_story))
    task_results[uuid] = {"status":"done","result":[generated_test_cases,similar_examples]}
    print("Model execution done!")

# prompt: create a sample flask ngork colab, and expose ngork url for public use
import uuid

from flask import Flask, request, jsonify
# from flask_ngrok import run_with_ngrok
from flask_cors import CORS
app = Flask(__name__)


CORS(app)

import time
import threading

@app.route("/ping")
def hello():
    return  jsonify({"Server says":"PONG!"})

@app.route("/generate", methods=["POST"])
def generate_test_cases_endpoint():
    """
    A POST endpoint to receive a prompt and generate test cases.
    Expects a JSON payload with a 'prompt' key.
    """
    if request.method == "OPTIONS":
        # Required: explicitly respond with HTTP 200 for preflight
        return '', 200

    try:
        if request.method == "POST":
           data = request.json
           uid = str(uuid.uuid4())

          #  generated_tc_output,similiar = generate_tc(uid,data['inp_user_story'])
          #  generated_tc_output = generate_fake_tc()
          #  similiar = fake_similiar
           print("Generating test cases...",uid,data['inp_user_story'])
           print("ARGS:", uid, data['inp_user_story'], type(data['inp_user_story']))

           thread = threading.Thread(target=generate_tc, args=(uid,data['inp_user_story']))
           print("Thread started")
           thread.deamon = True
           thread.start()
           return jsonify({"id":uid}), 202

            # if request.is_json:
            #     prompt = request.json.get("prompt") 

            #     if prompt is None:
            #         return jsonify({"error": "Missing 'prompt' in request body"}), 400

           
            # else:
            #     return jsonify({"error": "Content-Type must be application/json"}), 415
        else:
            return jsonify({"error": "Invalid request method"}), 405
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route("/result/<uuid>", methods=["GET"])
def get_result(uuid):
    print("Polled!")
    if uuid not in task_results:
        return jsonify({"error": "Task not found"}), 404
    
    task = task_results.get(uuid)
    print(task)
    if task :
      return jsonify({"status": task["status"], "result": task["result"]}), 200
    

def run_flask():
    app.run(port=5005)

def run_cloudflared():
    time.sleep(1)  # wait for Flask to start
    # !./cloudflared tunnel --url http://localhost:5000 > tunnel.log 2>&1 &

    # Wait for tunnel to come up
    while True:
        with open("tunnel.log", "r") as f:
            lines = f.read()
            if "trycloudflare.com" in lines:
                # clear_output(wait=True)
                print("🌍 Public URL:")
                for line in lines.split("\n"):
                    if "trycloudflare.com" in line:
                        print(line)
                        break
                break
        time.sleep(1)

# Run both in threads
threading.Thread(target=run_flask).start()
run_cloudflared()

# !./cloudflared tunnel --url http://localhost:5005
