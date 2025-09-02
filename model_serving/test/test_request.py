from mlserver.codecs import NumpyCodec
import requests
import numpy as np


inference_request = {
    "inputs": [
        NumpyCodec.encode_input(name='docs', payload=np.array([5.8, 2.8, 5.1, 2.4]), use_bytes=False).model_dump()
    ]
}


print(inference_request)

r = requests.post('http://0.0.0.0:8080/v2/models/ref-model/infer', json=inference_request)
print(r.json())
