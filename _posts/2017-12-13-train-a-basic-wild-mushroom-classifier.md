---
layout: post
title:  "Train a basic wild mushroom classifier"
date:   2017-12-13 14:30:00
excerpt: "(draft mode) Train a basic image classification model to identify 5 types of wild mushrooms, with Transfer Learning, Tensorflow, Tensorboard, MobileNet, and ImageNet images."
image: "/images/blog/mushroom-classifier-poc.png"
---

Earlier, we learnt about [transfer learning with Tensorflow for Poets retrain scripts](http://fungai.org/2017/12/11/tensorflow-for-poets/), and [downloading images from ImageNet](http://fungai.org/2017/12/11/tensorflow-for-poets/).

In this article we will focus on combining these concepts and techniques and build a more specialized machine learning application.

We are going to build a very basic image classification model to identify 5 types of wild mushrooms (we also show the WordNet ID `wnid` in bracket):

- Fly Agaric (n13003061)
- Scarlet Elf cup (n13030337)
- Common Stinkhorn (n13040629)
- Giant Puffball (n13044375)
- Earthstar (n13044778)

To do this, we will need to accomplish the following at a high level:

1. download reasonable amount of images per wild mushroom type from ImageNet. I'd say 250 is a good number.
2. ensure our images are clean. e.g. it needs to open up, in `.jpg` format, non corrupted, belong to correct category (I've seen non-mushroom objects in a mushroom category - this needs to be removed), non flickr dummy image. I tend to use image with at least 100 KB in file size for safety, but at least 20 KB. Anything less than 2KB is suspicious.
3. move 200 images per category to a folder structure suitable for the retrain script.
4. move the remaining 50 per category to a folder in same structure as above, but we'll use this for demo test running against the retrained model (i.e. no used for training).
5. run the retrain script `retrain.py` (with the appropriate options).
6. visualize training process with Tensorboard. This will give us an idea of model quality / hyperparameter tweaks required
7. test out the model by using the image inference / prediction script `label_image.py` (with the appropriate options).

### Step 1: Download 250 Images per Category

(work in progress)

### Step 2: Clean the 250 Images per Category

(work in progress)

### Step 3: Move 200 Images per category to a folder for image retraining

(work in progress)

### Step 4: Move 50 Images per category to a folder to perform demo inferences

(work in progress)

### Step 5: Perform Model Retraining (Transfer Learning)

#### How to use the retrain script

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

### Step 6: Visualize Retraining on Tensorboard

(work in progress)

### Step 7: Perform demo inferences

(work in progress)

## Summary


### Next Steps
