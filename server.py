from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from trainer import Trainer

app = FastAPI()

class TrainingConfig(BaseModel):
    model_name: str
    num_epochs: int
    batch_size: int
    learning_rate: float

@app.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    trainer = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start":
                config = message["config"]
                print(f"Starting training with config: {config}")
                
                async def progress_callback(stats):
                    await websocket.send_json({
                        "type": "progress",
                        "data": stats
                    })
                
                trainer = Trainer(config, callback=progress_callback)
                await trainer.train()
                
                await websocket.send_json({"type": "finished"})
                
            elif message["type"] == "stop":
                if trainer:
                    trainer.stop()
                    await websocket.send_json({"type": "stopped"})
                    
    except WebSocketDisconnect:
        print("Client disconnected")
        if trainer:
            trainer.stop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
