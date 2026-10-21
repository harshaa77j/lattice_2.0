import os
import re
import socket
import sys
from dotenv import load_dotenv
from shared.protocol import NODE_ID_DISALLOWED, NODE_ID_MAX_LEN, is_valid_node_id, NOT_REGISTERED
import requests
import time
import threading

REQUEST_TIMEOUT = 5 
REGISTER_RETRY_DELAY = 3
HEARTBEAT_INTERVAL = 5 

def node_id_from_hostname():
    name = socket.gethostname().split(".")[0] 
    name = re.sub(NODE_ID_DISALLOWED, "-", name)
    return name[:NODE_ID_MAX_LEN].strip("-_") 

def fail(message):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)

def load_config():
    load_dotenv()
    
    server = (os.getenv("LATTICE_SERVER") or "").strip().rstrip("/")
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
    node_to_register={
        "node_id": config['node_id'],
        "cpu_count": config['cpu_count'],
        "device": config['device']
    }
    while True:
        try:
            response=requests.post(endpoint_url, json=node_to_register, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("Registration success!")
                return True
            if 400 <= response.status_code < 500:
                raise Rejected(f"Server rejected registration ({response.status_code}): {response.text}")
            print(f"Server returned status code: {response.status_code}, message: {response.text}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"Network error encountered: {e.__class__.__name__}", file=sys.stderr)
        print(f"Retrying registration in {REGISTER_RETRY_DELAY} seconds...\n")
        time.sleep(REGISTER_RETRY_DELAY)
  
class Rejected(Exception):
    """Raised when the server rejects registration (4xx error)."""
    
def is_not_registered(response):
    try:
        data=response.json()
        if isinstance(data, dict):
            return data.get("detail") == NOT_REGISTERED
        return False
    except ValueError:
        return False
      
def heartbeat_loop(config):
    endpoint_url = f"{config['server']}/heartbeat/{config['node_id']}"
    while True:
        try:
            response=requests.post(endpoint_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("heartbeat ok")
            elif response.status_code == 404 and is_not_registered(response):
                try:
                    print(f"{NOT_REGISTERED} re-registering...")
                    register(config)
                except Rejected as e:
                    print(f"[WARNING] Registration rejected by server: {e}. Continuing...")
                    time.sleep(60)
            else:
                print(f"Server returned status code: {response.status_code}, message: {response.text}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"Network error encountered: {e.__class__.__name__}", file=sys.stderr)
        except Exception as e:
            print(f"Heartbeat: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        time.sleep(HEARTBEAT_INTERVAL)
 
def main():
    config = load_config()
    register(config)
    heartbeat_thread = threading.Thread(
        daemon=True,  
        target=heartbeat_loop,  
        args=(config,),                    
        name="heartbeat",
        )
    heartbeat_thread.start()   
    try:
        while True:
            print("main: pretending to train for 60 s")
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping node client.")

if __name__ == "__main__":
    main()