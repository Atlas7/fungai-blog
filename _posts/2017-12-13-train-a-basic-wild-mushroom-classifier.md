---
layout: post
title:  "Train a basic wild mushroom classifier"
date:   2017-12-13 14:30:00
excerpt: "(draft mode) Train a basic image classification model to identify 5 types of wild mushrooms, with Transfer Learning, Tensorflow, Tensorboard, MobileNet, and ImageNet images."
image: "/images/blog/mushroom-classifier-poc.png"
---

Earlier, we learnt about [transfer learning with Tensorflow for Poets retrain scripts](http://fungai.org/2017/12/11/tensorflow-for-poets/), and [how to download images from ImageNet](http://fungai.org/2017/12/11/tensorflow-for-poets/). In this article we will focus on combining these concepts and techniques and build a more specialized machine learning application.

### Build a Basic Wild Mushroom Classifier

In this tutorial, we are going to build a very basic image classification model together, to identify 5 types of wild mushrooms

- Fly Agaric ([view on ImageNet: n13003061](http://www.image-net.org/synset?wnid=n13003061))
- Scarlet Elf cup ([view on ImageNet: n13030337](http://www.image-net.org/synset?wnid=n13030337))
- Common Stinkhorn ([view on ImageNet: n13040629](http://www.image-net.org/synset?wnid=n13040629))
- Giant Puffball ([view on ImageNet: n13044375](http://www.image-net.org/synset?wnid=n13044375))
- Earthstar ([view on ImageNet: n13044778](http://www.image-net.org/synset?wnid=n13044778))

But before getting our hands dirty let's take a step back and form our high level strategy. We'll use Google's 7 Steps of Machine Learning to guide our implementation process.

### Google's 7 Steps of Machine Learning

Let's review quickkly the Google's [7 Steps of Machine Learning](https://www.youtube.com/watch?v=nKW8Ndu7Mjw) (see 9:38 - 9:53):


<iframe width="560" height="315" src="https://www.youtube.com/embed/nKW8Ndu7Mjw?rel=0&amp;start=578" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen></iframe>


Expand these 7 steps to suit our Wild Mushroom Classifier Project:

1. Gathering Data
  - Download reasonable amount of labelled images per wild mushroom type from ImageNet.
  - We'll need at least 250 labelled images per category: 200 for retraining (80% train, 10% validation, 10% test), and 50 for demo predictions later.
  - In other words, we will have at least 1250 labelled images for retraining (5 categories x 200 per category), and 250 labelled images for demo predictions (5 categories x 50 per category).
  - In total we will have 1500 images (1250 for retraining, and 250 for demo).
  - We'll use [ImageNet_Utils](https://github.com/tzutalin/ImageNet_Utils) to help us download labelled images from ImageNet easily. Note that we'll likely download more than we need to begin with. But this is ok, as we'll only pick what we need in the data preparation phase in step 2 (250 labelled images per category).
  - We'll now have 5 new folders containing our 5 categories of images.
2. Preparing that Data
  - Now that we've downloaded many images from ImageNet, we'll manually pick 250 images per category and copy into a new directory structure (say, to a folder called `shrooms_250_clean`). This will also help us avoid data imbalances, as we'll have equal amount of images per category.
  - Do the image cleansing in our newly created `shrooms_250_clean`. e.g. `.jpg` format, non corrupted, correct category, non flickr dummy image, reasonable file size, etc. Delete as appripriate.
  - As we do the image delete we may fall short on the 250 images per category target. Just copy any unused images from the raw directories (at end of step 1 as needed)
3. Choosing a Model
  - We will use [Tensorflow for poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) as our starting point baseline
  - We have two obvious choices: Inception v3 and MobileNet.
  - Inception v3 is more accurate but heavier.
  - MobileNet is slightly less accurate but lighter and more suitable for low-power embedded devices.
  - Since this is our first attempt, let's go for something light. We will use the MobileNet model.
  - (optional) at some later point, once we become more familiar with the process, we can try out other open source models out there on the internet.
4. Training
  - Run the [Tensorflow for poets retrain script](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3) `retrain.py` with the appropriate options.
  - Use tensorboard to monitor training progress and performance (mainly accuracy and cross entropy charts)
5. Evaluation 
  - Again, use tensorboard to monitor training progress and performance (mainly accuracy and cross entropy charts)
6. Hyperparameter Tuning
  - Can we improve our training accuracy (higher the better) and Cross Entropy Error (lower the better)?
  - Again, use Tensorboard.
7. Prediction
  - Remember our set-aside 50 demo images (per category) that were not used for training? Let's use our now trained model to perform prediction (aka inference) with the help of [Tensorflow for poets prediction script](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3)  `label_image.py`, with appropriate options.
  - Of the 250 demo images (5 category x 50 images per category), how many did the model predict it correctly?
  - For the ones that the model predicted wrong, what do the images look like? This will give us an idea of whether the error is reasonable. For instance, if even a human would have a hard time classifying the image, maybe the model is not doing that bad.
  - (optional) create a program that automatically display images one by one (or in small batches), and perform prediction (and compute overall accuracy and error rate) in the same time? This may be good for showing off / demo purpose?

### KISS - Keep it Simple, Stupid

Our aim is to get things working as quickly as possible - understanding just enough about the high level process and implementing the workflow and get to the end. By walking through the process we will gain an appreciation on how we may potentially improve the process a bit. But that will be another job for another day. For now, let's [keep it simple and stupid](https://en.wikipedia.org/wiki/KISS_principle): we will complete an iteration loop as quickly as possible, from start to finish, and see some end results. (and we will, I promise!)

OK, buckle up. Here we go.

### Directory Structure

For ease of explaining, i'm going to assume any Github repositories are stored at `~/repos`. i.e. our directory structure will eventually look like this:

```
|- repos
  |- github-repo-1
  |- github-repo-2
  |- etc.
```

If you do not have a `repos` folder in your home directory, create it now:

```
$ mkdir ~/repos
$ cd ~
```

We will need two repositories in our `repos`:

```
|- repos
  |- my-ImageNet_Utils
  |- my-tensorflow-for-poets
```

The two repositories above are actually from Github. You can download (and rename) these two repositories to your mac locally by doing this:

```
$ cd ~
$ cd repos
$ git clone https://github.com/tzutalin/ImageNet_Utils my-ImageNet_Utils
$ git clone https://github.com/googlecodelabs/tensorflow-for-poets-2 my-tensorflow-for-poets
```

Notice that I've renamed the original Github repositories `ImageNet_Utils` to `my-ImageNet_Utils`, and `tensorflow-for-poets-2` to `my-tensorflow-for-poets`. This is purely to avoid potential clashes or confusion if you happen already have `ImageNet_Utils` and `tensorflow-for-poets-2` setup on your mac. 

In this article however, I'm going to stick with `my-ImageNet_Utils` and `my-tensorflow-for-poets`, purely for instructional convenience. (In reality however you can name the repositories to whatever name you'd like.)

### Step 1: Gathering Data

We download images from ImageNet with the help of this handy [ImageNet_Utils tool](https://github.com/tzutalin/ImageNet_Utils), as introduced in [a previous article]({% post_url 2017-12-12-download-imagenet-images-by-wnid %}). I'm going to recite the exact steps here:

Navigate to `my-ImageNet_Utils`repository directory (so we can see the Python files if we want to):

```
$ cd ~/repos/my-ImageNet_Utils
```

Ensure we have Python 2.7 environment setup (yes I know. We are in this age we should be using Python 3.6+ really. But from what I've tested so far, Python 2.7 works for this ImageNet_Utils tool. So just stick with it for now. We are only ever going to use it strictly for downloading ImageNet images). Let's create a Python 2.7 environment with Anaconda (if you've already done this, skip to next step)

```
$ conda create --name py27p13 python=2.7
```

Activate the Python environment (Python 2.7.13 is what I use):

```
$ source activate py27p13
```

We should see our conda environment name in our prompt like this:

```
(py27p13) $
```

Now the fun part. We are going to download all ImageNet images for the 5 mushroom categories, as represented by the WordNet ID `wnid`.

Open a terminal and start downloading all the images of Fly Agaric:

```
(py27p13) $ cd ~/repos/my-ImageNet_Utils
(py27p13) $ ./downloadutils.py --downloadImages --wnid n13003061
```

This will download the images in a new sub-directory (within the `ImageNet_Utils` repository) at:

```
~/repos/ImageNet_Utils/13003061/n13003061_urlimages
```

There will be some minor errors during the download process, due to:

- URL no longer valid
- URL points to a website instead of an image
- etc

You can safetly ignore those errors. Of the many image URLs, we will at least get 250 (hopefully) good images, or more.

Do the same for the other `wnid`:


```
(py27p13) $ cd ~/repos/my-ImageNet_Utils
(py27p13) $ ./downloadutils.py --downloadImages --wnid n13030337
(py27p13) $ ./downloadutils.py --downloadImages --wnid n13040629
(py27p13) $ ./downloadutils.py --downloadImages --wnid n13044375
(py27p13) $ ./downloadutils.py --downloadImages --wnid n13003061
```

Once all is done we shall see our 5 new directories at `~/repos/my-ImageNet_Utils` :

![imagenet-download-1.png](/images/blog/imagenet-download-1.png)

Let's take a look at the downloaded images:

Fly Agaric images at `~/repos/ImageNet_Utils/n13030337/n13003061_urlimages`:

![imagenet-download-2.png](/images/blog/imagenet-download-2.png)

Scarlet Elf cup images at `~/repos/ImageNet_Utils/n13030337/n13030337_urlimages`:

![imagenet-download-3.png](/images/blog/imagenet-download-3.png)

Common Stinkhorn images at `~/repos/ImageNet_Utils/n13040629/n13040629_urlimages`:

![imagenet-download-4.png](/images/blog/imagenet-download-4.png)

Giant Puffball images at `~/repos/ImageNet_Utils/n13044375/n13044375_urlimages`:

![imagenet-download-5.png](/images/blog/imagenet-download-5.png)

Earthstar images at `~/repos/ImageNet_Utils/n13003061/n13003061_urlimages`:

![imagenet-download-6.png](/images/blog/imagenet-download-6.png)

Notice that we will inevitably have more images in certain categories than the other from our initial raw datasets.



### Step 2: Preparing that Data

(work in progress)

### Step 3: Choosing a Model

(work in progress)

### Step 4: Training

(work in progress)

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

### Step 5: Evaluation

### Step 6: Hyperparameter Tuning

(work in progress)

### Step 7: Prediction

(work in progress)

### Summary

(work in progress)

### Next Steps

(work in progress)
