# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

from predict_client.prod_client import ProdClient

# <codecell>

client_1 = ProdClient('localhost:9000', 'model_sub', 1)
client_2 = ProdClient('localhost:9000', 'model_add', 4)

# <codecell>

req_data = [{'in_tensor_name': 'input', 'in_tensor_dtype': 'DT_INT32', 'data': 5}]
print(client_1.predict(req_data))

req_data = [{'in_tensor_name': 'input', 'in_tensor_dtype': 'DT_INT32', 'data': 5}]
print(client_2.predict(req_data))
