#Rules the server and nodes must agree on

import re

NODE_TOKEN_HEADER = "X-Lattice-Token"        # sent by volunteer nodes
RESEARCHER_KEY_HEADER = "X-Researcher-Key"   # sent by the researcher website
ADMIN_KEY_HEADER = "X-Admin-Key" 
NODE_ID_MAX_LEN = 64
NODE_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
_NODE_ID_RE = re.compile(NODE_ID_PATTERN)
NODE_ID_DISALLOWED = r"[^A-Za-z0-9_-]"
NOT_REGISTERED="Node not Registered!"
DATA_SEED = 42

def is_valid_node_id(node_id: str) -> bool:
    return 0 < len(node_id) <= NODE_ID_MAX_LEN and _NODE_ID_RE.fullmatch(node_id) is not None