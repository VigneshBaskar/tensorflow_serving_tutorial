# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import os
import tensorflow as tf

# <codecell>

# Graph

placeholder_name = 'a'
operation_name = 'add'

a = tf.placeholder(tf.int32, name=placeholder_name)
b = tf.constant(10)

c = tf.add(a, b, name=operation_name)

# <codecell>

with tf.Session() as sess:
    ten_plus_two = sess.run(c, feed_dict={a: 2})


# <codecell>

export_path_base = 'Model_A'
model_version = '4'
export_path = os.path.join(export_path_base, model_version)

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_a = tf.saved_model.utils.build_tensor_info(a)
tensor_info_c = tf.saved_model.utils.build_tensor_info(c)

# <codecell>

prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
  inputs={'input': tensor_info_a},
  outputs={'output': tensor_info_c},
  method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

# <codecell>

builder.add_meta_graph_and_variables(
    sess,
    [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        'serving_default':
            prediction_signature
    }
) 
builder.save()
