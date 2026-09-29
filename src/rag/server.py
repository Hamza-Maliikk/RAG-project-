import json
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from rag.pipeline import rag_answer

app = FastAPI()


def make_chunk(model, content=None, finish=None):
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish,
        }],
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model", "my-rag")
    messages = body.get("messages", [])

    question = ""
    for m in reversed(messages):
        if m["role"] == "user":
            c = m["content"]
            question = c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in c if p.get("type") == "text")
            break

    answer = rag_answer(question)

    if not body.get("stream", False):
        return JSONResponse({
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
        })

    def stream():
        for word in answer.split(" "):
            yield f"data: {json.dumps(make_chunk(model, word + ' '))}\n\n"
        yield f"data: {json.dumps(make_chunk(model, finish='stop'))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")