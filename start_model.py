import sys
import json
from dd_client import DD

with open('config.json', 'r') as f:
    config = json.load(f)

service = sys.argv[1]

MODEL_REPO = config['REPO'][service]['PATH']
nclasses = config['REPO'][service]['CLASS_COUNT']
height = width = 224

# setting up DD client
host = 'localhost'
sname = config['REPO'][service]['NAME']
description = config['REPO'][service]['DESCRIPTION']
mllib = 'caffe'
dd = DD(host)
dd.set_return_format(dd.RETURN_PYTHON)

# creating ML service
model = {'repository': MODEL_REPO}
parameters_input = {'connector': 'image', 'width': width, 'height': width}
parameters_mllib = {'nclasses': nclasses}
parameters_output = {}

dd.put_service(sname,
               model,
               description,
               mllib,
               parameters_input,
               parameters_mllib,
               parameters_output)
