import uuid
import threading
from flask import Blueprint, request, jsonify
from .tasks import generate_tc, task_results

api = Blueprint("api", __name__)

@api.route("/ping")
def ping():
    return jsonify({"Server says": "PONG!"})

@api.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        uid = str(uuid.uuid4())
        thread = threading.Thread(target=generate_tc, args=(uid, data['inp_user_story']))
        thread.daemon = True
        thread.start()
        return jsonify({"id": uid}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route("/result/<uuid>", methods=["GET"])
def result(uuid):
    if uuid not in task_results:
        return jsonify({"error": "Task not found"}), 404
    task = task_results[uuid]
    return jsonify({"status": task["status"], "result": task["result"]}), 200
