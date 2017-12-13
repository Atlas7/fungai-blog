---
layout: post
title:  "Train a basic wild mushroom classifier"
date:   2017-12-13 14:30:00
excerpt: "Train a basic image classification model to identify 5 types of wild mushrooms, with Transfer Learning, Tensorflow, Tensorboard, MobileNet, and ImageNet images."
image: "/images/blog/mushroom-classifier-poc.png"
---

### How to use the retrain script

To see what options are there, do a: 

```
$ python -m scripts.retrain -h
```

Usage:

```
$ python -m scripts.retrain [--option value]
```

For example:

```
$ python -m scripts.retrain \
  --image_dir=tf_files/${TFP_IMAGES_DIR}
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${TFP_ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${TFP_ARCHITECTURE}" \
```

#### Retrain options

Instead of having to issue `help` every time, I've just had a read through the code and manually document the options here for better understanding / ease of reference.

```
| option                            | type  | default value              | description                                                                 |
|-----------------------------------|-------|----------------------------|-----------------------------------------------------------------------------|
| image_dir                         | str   | ""                         | Path to folders of labeled images.                                          |
| output_graph                      | str   | "/tmp/output_graph.pb"     | Where to save the trained graph.                                            |
| intermediate_output_graphs_dir    | str   | "/tmp/intermediate_graph/" | Where to save the intermediate graphs.                                      |
| intermediate_store_frequency      | int   | 0                          | How many steps to store intermediate graph. If "0" then will not store.     |
| output_labels                     | str   | "/tmp/output_labels.txt"   | Where to save the trained graph's labels.                                   |
| summaries_dir                     | str   | "/tmp/retrain_logs"        | Where to save summary logs for TensorBoard.                                 |
| how_many_training_steps           | float | 0.01                       | How large a learning rate to use when training.                             |
| testing_percentage                | int   | 10                         | What percentage of images to use as a test set.                             |
| validation_percentage             | int   | 10                         | What percentage of images to use as a validation set.                       |
| eval_step_interval                | int   | 10                         | How often to evaluate the training results.                                 |
| train_batch_size                  | int   | 100                        | How many images to train on at a time.                                      |
| test_batch_size                   | int   | -1                         | How many images to test on. This test set is only used once, to evaluate    |
|                                   |       |                            |  the final accuracy of the model after training completes.                  |
|                                   |       |                            |  A value of -1 causes the entire test set to be used, which leads to more   |
|                                   |       |                            |  stable results across runs.                                                |
| validation_batch_size             | int   | 100                        | How many images to use in an evaluation batch. This validation set is       |
|                                   |       |                            |  used much more often than the test set, and is an early indicator of how   |
|                                   |       |                            |  accurate the model is during training.                                     |
|                                   |       |                            |  A value of -1 causes the entire validation set to be used, which leads to  |
|                                   |       |                            |  more stable results across training iterations, but may be slower on large |
|                                   |       |                            |  training sets.                                                             |
| print_misclassified_test_images   | bool  | False                      | Whether to print out a list of all misclassified test images.               |
| model_dir                         | str   | "/tmp/imagenet"            | Path to classify_image_graph_def.pb, imagenet_synset_to_human_label_map.txt |
|                                   |       |                            |  , and imagenet_2012_challenge_label_map_proto.pbtxt.                       |
| bottleneck_dir                    | str   | "/tmp/bottleneck"          | Path to cache bottleneck layer values as files.                             |
| final_tensor_name                 | str   | "final_result"             | The name of the output classification layer in the retrained graph.         |
```
