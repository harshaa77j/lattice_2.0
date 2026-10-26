import os
import re
import socket
import sys
from dotenv import load_dotenv
from shared.protocol import NODE_ID_DISALLOWED, NODE_ID_MAX_LEN, is_valid_node_id, NOT_REGISTERED, NODE_TOKEN_HEADER
import requests
import time
import threading
import logging

REQUEST_TIMEOUT = 5 
REGISTER_RETRY_DELAY = 3
HEARTBEAT_INTERVAL = 5 
log = logging.getLogger(__name__)

def node_id_from_hostname():
    name = socket.gethostname().split(".")[0] 
    name = re.sub(NODE_ID_DISALLOWED, "-", name)
    return name[:NODE_ID_MAX_LEN].strip("-_") 

def fail(message):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)

class Rejected(Exception):
    """Raised when the server rejects registration (4xx error)."""
    
def load_config():
    load_dotenv()
    
    node_token = os.getenv("LATTICE_TOKEN", "").strip()
    if not node_token:
        fail("LATTICE_TOKEN is not set. Add it to node/.env ")
    
    server = (os.getenv("LATTICE_SERVER") or "").strip().rstrip("/")
    if not server:
        fail("LATTICE_SERVER is missing or empty. Set it in node/.env.")
    if not server.startswith(("http://", "https://")):
        fail(f"LATTICE_SERVER must start with http:// or https:// (got '{server}').")

    node_id=(os.getenv("NODE_ID") or "").strip()
    if not node_id:
        node_id = node_id_from_hostname()
        log.warning(f"NODE_ID not set. Using hostname: '{node_id}'")
    if not is_valid_node_id(node_id):
        fail(f"node id '{node_id}' is invalid. Use letters, digits, - and _ only "
             f"(max {NODE_ID_MAX_LEN} characters). Set NODE_ID in node/.env.")

    return {
        "server": server,
        "token" : node_token,
        "node_id": node_id,
        "cpu_count": os.cpu_count() or 1,
        "device": "cpu"
    }
    
def make_session(config):
    session = requests.Session()
    session.headers.update({NODE_TOKEN_HEADER: config["token"]})
    return session  

def register(config,session):
    endpoint_url = f"{config['server']}/register"
    node_to_register={
        "node_id": config['node_id'],
        "cpu_count": config['cpu_count'],
        "device": config['device']
    }
    while True:
        try:
            response=session.post(endpoint_url, json=node_to_register, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                log.info(f"Registration success for {node_to_register['node_id']} to server {config['server']}")
                return True
            if 400 <= response.status_code < 500:
                raise Rejected(f"Server rejected registration ({response.status_code}): {response.text}")
            log.warning(f"Server returned status code: {response.status_code}, message: {response.text}")
        except requests.RequestException as e:
            log.warning(f"Network error encountered: {e.__class__.__name__}")
        log.info(f"Retrying registration in {REGISTER_RETRY_DELAY} seconds...")
        time.sleep(REGISTER_RETRY_DELAY)
  
def is_not_registered(response):
    try:
        data=response.json()
        if isinstance(data, dict):
            return data.get("detail") == NOT_REGISTERED
        return False
    except ValueError:
        return False
      
def heartbeat_loop(config,session):
    endpoint_url = f"{config['server']}/heartbeat/{config['node_id']}"
    while True:
        try:
            response=session.post(endpoint_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                log.debug("heartbeat ok")
            elif response.status_code == 404 and is_not_registered(response):
                try:
                    log.info(f"{NOT_REGISTERED} re-registering...")
                    register(config,session)
                except Rejected as e:
                    log.error(f"Registration rejected by server: {e}. Continuing...")
                    time.sleep(60)
            else:
                log.warning(f"Server returned status code: {response.status_code}, message: {response.text}")
        except requests.RequestException as e:
            log.warning(f"Network error encountered: {e.__class__.__name__}")
        except Exception as e:
            log.exception(f"Heartbeat: unexpected error {type(e).__name__}: {e}")
        time.sleep(HEARTBEAT_INTERVAL)
 
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    config = load_config()
    try:
        try:
            register_session=make_session(config)
            register(config,register_session)
        except Rejected as e:
            fail(str(e))
        heartbeat_session=make_session(config)
        heartbeat_thread = threading.Thread(
            daemon=True,  
            target=heartbeat_loop,  
            args=(config,heartbeat_session),                    
            name="heartbeat",
            )
        heartbeat_thread.start()   
        while True:
            log.info("main: pretending to train for 60 s")
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("Stopping node client.")

if __name__ == "__main__":
    main()