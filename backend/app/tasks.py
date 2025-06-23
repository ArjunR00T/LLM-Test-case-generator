from .similarity import get_similar_examples
from .model import run_inference
from .utils import clean_and_parse_partial_json

task_results = {}

def generate_tc(uuid, new_story):
    task_results[uuid] = {"status": "processing", "result": None}
    similar = get_similar_examples(new_story)
    output = run_inference(json.dumps(similar), new_story)
    parsed = clean_and_parse_partial_json(output)
    task_results[uuid] = {"status": "done", "result": [parsed, similar]}
