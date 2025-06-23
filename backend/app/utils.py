import json
import re

def clean_and_parse_partial_json(raw_output):
    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    cleaned = re.sub(r"```json|```", "", raw_output.strip())
    match = re.search(r"\[\s*{.*", cleaned, re.DOTALL)
    if not match:
        return None

    json_str = match.group()
    try:
        return json.loads(json_str)
    except:
        try:
            json_str = re.sub(r",\s*{[^}]*$", "", json_str.strip()) + "]"
            return json.loads(json_str)
        except Exception:
            return None
