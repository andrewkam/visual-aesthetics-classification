import sys
import json
from dd_client import DD

with open('config.json', 'r') as f:
    config = json.load(f)

service = sys.argv[1]

# setting up DD client
host = 'localhost'
sname = config['REPO'][service]['NAME']
dd = DD(host)
dd.set_return_format(dd.RETURN_PYTHON)

dd.delete_service(sname, 'full')
