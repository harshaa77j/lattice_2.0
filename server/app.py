from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime
import time

HEARTBEAT_TIMEOUT=20

app=FastAPI()

@app.get("/")
def check():
    return {"message":"working"}

class NodeRegistration(BaseModel):
    node_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    cpu_count: int = Field(gt=0)  
    device: Literal['cuda','cpu','mps'] 
    
class NodeRecord(NodeRegistration):
    last_seen: datetime = Field(default_factory=datetime.now)
    last_seen_mono: float = Field(default_factory=time.monotonic, exclude=True)
    
#in-mem
nodes: dict[str, NodeRecord] = {}

@app.post("/register")
def register_node(node: NodeRegistration):
    data=node.model_dump()
    record=NodeRecord(**data)
    nodes[node.node_id]=record
    return record
    
@app.post("/heartbeat/{node_id}")
def heartbeat(node_id:str ):
    node = nodes.get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not Registered!")
    node.last_seen=datetime.now()
    node.last_seen_mono = time.monotonic() 
    return {"ok": True}

@app.get("/status")
def status():
    stats={}
    now = time.monotonic()
    for node_id,record in nodes.items():
        node=record.model_dump()
        elapsed = now-record.last_seen_mono
        node["alive"]=elapsed <= HEARTBEAT_TIMEOUT
        stats[node_id]=node
    return stats