import os
import re
import socket
import sys
from dotenv import load_dotenv
from shared.protocol import NODE_ID_DISALLOWED, NODE_ID_MAX_LEN, is_valid_node_id
import requests

REQUEST_TIMEOUT = 5 

def node_id_from_hostname():
    name = socket.gethostname().split(".")[0] 
    name = re.sub(NODE_ID_DISALLOWED, "-", name)
    return name[:NODE_ID_MAX_LEN].strip("-_") 

def fail(message):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)

def load_config():
    load_dotenv()
    
    server=os.getenv("LATTICE_SERVER")
    if not server:
        fail("LATTICE_SERVER is missing or empty. Set it in node/.env.")
    if not server.startswith(("http://", "https://")):
        fail(f"LATTICE_SERVER must start with http:// or https:// (got '{server}').")

    node_id=(os.getenv("NODE_ID") or "").strip()
    if not node_id:
        node_id = node_id_from_hostname()
        print(f"Warning: NODE_ID not set. Using hostname: '{node_id}'")
    if not is_valid_node_id(node_id):
        fail(f"node id '{node_id}' is invalid. Use letters, digits, - and _ only "
             f"(max {NODE_ID_MAX_LEN} characters). Set NODE_ID in node/.env.")

    return {
        "server": server,
        "node_id": node_id,
        "cpu_count": os.cpu_count() or 1,
        "device": "cpu"
    }

def register(config):
    
    endpoint_url = f"{config['server']}/register"
    node_to_register=[{
        "node_id": config['node_id'],
        "cpu_count": config['cpu_count'],
        "device": config['device']
    }]
    response=requests.post(endpoint_url, json=node_to_register, timeout=REQUEST_TIMEOUT)
    if response.status_code == 200:
        print("success")
        return True
    else:
        print(response.status_code)
        print(response.text)
        return False
    
def main():
    config = load_config()
    register(config)

if __name__ == "__main__":
    main()