# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

from predict_client.prod_client import ProdClient

# <codecell>

client = ProdClient('localhost:9000', 'model_add', 4)

# <codecell>

req_data = [{'in_tensor_name': 'input', 'in_tensor_dtype': 'DT_INT32', 'data': 5}]

# <codecell>

client.predict(req_data)
