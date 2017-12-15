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

Let's review quickly the Google's [7 Steps of Machine Learning](https://www.youtube.com/watch?v=nKW8Ndu7Mjw) (see 9:38 - 9:53):


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
$ conda create --name py27p13 python=2.7.13
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

#### Decision: Pick our subset and clean (or the other way round?)

Now that we have the raw ImageNet images downloaded to `~/repos/ImageNet_Utils/`, we have a decision to make: we can either (1) clean all the raw images, then pick our random 250 images per category, or (2) the other way round - pick aound 250 images per category, then clean this smaller subset. Let's compare these two options.

Option (1): clean everything up front once and then pick our 250 images per category. Advantage of this is that know our entire dataset will be clean at the end of the data cleansing. This allow flexibility in long run - for instance, we will be able to select our fixed subset of any size, be it 250 images per category, 300 per category, or even 500 per category, we will be able to do that easily. The only drawback of this option is the massive effort in cleaning more data than we actually need upfront, for our initial prototype. If you have already done the tensorflow for poets tutorial previously, you'll know that around 200 retrain images will be good enough to get started. If we look at fly agaric alone, we already have 850 ish images in this category - to clean all 850 fly agaric images when all we need is just 250 from that category, could slow down our first attempt to this exercise. Imagine we have 5 categories to clean! At the time of writing this, step 1 would have downloaded 842 Fly agaric images, 310 scarlet elf cup, 613 common stinkhorn, 643 giant puffball, and 572 earthstar. That's a total of 2980 images to clean (when all we need is 1250 images: 250 per category x 5 categories). This option will likely double the number of images to clean than actually required.

Option (2): pick our 250 images per category, and then clean these subsets. Advantage of this is **focus**. We know 250 clean images per category is good enough. We pick only 250 raw images and focus on getting them cleansed. This will ensure we are not distracted too much upfront, at such an early stage. The downside of this option is that (as you've probably have guessed), is that whenever we see invalid images, we delete them. So say we have started with 250 fly agaric images, and we've found 30 dirty ones and ended up deleting them and reduce our bucket size to 220 fly agaric images. We then copy and paste the additional 30 unused fly agaric images to this bucket (to make it up to 250). We do the cleaning on this 30 images, find 5 dirty ones, delete them, and ended up with 245 clean images in this bucket. We repeat the process until we obtain the entire set of 250 clean fly agaric images. The downside as you see is the additional manual work involved. But with a systematic approach, it is possible to reduce the likelihood of dirty images in our 250 buckets. This option is not perfect, but for a first attempt in building this wild mushroom classificaton app, it is good enough for getting things done and gaining relevant experience, and so we will choose this option in this tutorial.

In this tutorial we will use option (2) - pick around 250 images per category, then clean this smaller subset.

#### Create a new folder dedicated to our 250 cleansed images per category

let's create a new folder to store our ~250 clean images per categories:

```
(py27p13) $ cd ~/repos/my-ImageNet_Utils
(py27p13) $ mkdir shrooms-clean-250-each
```

Within this `shrooms-clean-250-each`, let's create 5 subdirectories with appropriate names:

```
(py27p13) $ cd ~/repos/my-ImageNet_Utils
(py27p13) $ mkdir n13003061-fly-agaric
(py27p13) $ mkdir n13030337-scarlet-elf-cup
(py27p13) $ mkdir n13040629-common-stinkhorn
(py27p13) $ mkdir n13044375-giant-puffball
(py27p13) $ mkdir n13044778-Earthstar
```

Our directory `~/repos/my-ImageNet_Utils/shrooms-clean-250-each` should look like this:

![shrooms-clean-250-each](/images/blog/shrooms-clean-250-each.png)

#### Copy 250 images per category for cleansing

This step is a bit manual, but easy to do for first timer. Open up two Finder windows:

- Finder 1 points to the clean images directory: `~/repos/my-ImageNet_Utils`
- Finder 2 points to the retraining images directory: `~/repos/my-ImageNet_Utils/shrooms-clean-250-each` 

In Finder 1, sort the images by descending by file size. Copy the first 250 clean images per category over to Finder 2, like this:

<div class="table-wrapper" markdown="block">

| Category          | Copy from Finder 1                 | Paste to Finder 2               |
|-------------------|------------------------------------|---------------------------------|
| Fly Agaric        | `./n13030337/n13003061_urlimages/` | `./n13003061-fly-agaric/`       |
| Scarlet Elf cup   | `./n13030337/n13030337_urlimages/` | `./n13030337-scarlet-elf-cup/`  |
| Common Stinkhorn  | `./n13040629/n13040629_urlimages/` | `./n13040629-common-stinkhorn/` |
| Giant Puffball    | `./n13044375/n13044375_urlimages/` | `./n13044375-giant-puffball/`   |
| Earthstar         | `./n13003061/n13003061_urlimages/` | `./n13003061-earthstar/`        |

</div>

What is the reason for doing a sort descending by file size? It's just a trick: images with larger file size is likely to be more valid than smaller ones. In particular, images with 2KB or below are very likely invalid images. Focusing larger size images will likely reduce the population of "dirty" images. But it's entirely up to you how you'd like to do it.

#### Do some cleaning

Now, review the images in `~/repos/my-ImageNet_Utils/shrooms-clean-250-each/`. Delete any invalid images as appropriate.

For example, while I was checking through my Earthstar images, these are what I consider valid and invalid images:

##### Example: Valid Earth Star (keep these) <i class="fa fa-check" aria-hidden="true" />

<div class="container">
  <div class="row">
    <div class="col-sm-6"><img alt="valid-earthstar-1.png.png" src="/images/blog/valid-earthstar-1.png"/></div>
    <div class="col-sm-6"><img alt="valid-earthstar-2.png.png" src="/images/blog/valid-earthstar-2.png"/></div>
  </div>
  <div class="row">
    <div class="col-sm-6"><img alt="valid-earthstar-3.png.png" src="/images/blog/valid-earthstar-3.png"/></div>
    <div class="col-sm-6"><img alt="valid-earthstar-4.png.png" src="/images/blog/valid-earthstar-4.png"/></div>
  </div>
</div>

##### Example: Invnalid Earth Star (delete these) <i class="fa fa-times" aria-hidden="true" />

<div class="container">
  <div class="row">
    <div class="col-sm-6"><img alt="invalid-earthstar-1.png.png" src="/images/blog/invalid-earthstar-1.png"/></div>
    <div class="col-sm-6"><img alt="invalid-earthstar-2.png.png" src="/images/blog/invalid-earthstar-2.png"/></div>
  </div>
  <div class="row">
    <div class="col-sm-6"><img alt="invalid-earthstar-3.png.png" src="/images/blog/invalid-earthstar-3.png"/></div>
    <div class="col-sm-6"><img alt="imagenet-partial-downloaded-image.png" src="/images/blog/imagenet-partial-downloaded-image.png"/></div>
  </div>  
</div>

Notice that as we delete images, our "250 per bucket" will start to fall short. In this case, just add more unused images and repeat the cleaning step (probably a few times). In the end of we should have 250 clean images per category, stored at `~/repos/my-ImageNet_Utils/shrooms-clean-250-each/`.

#### Prepare the clean data for tensorflow for poets

First of all, recall our directory structure:

```
|- repos
  |- my-ImageNet_Utils
  |- my-tensorflow-for-poets
```

So far we have been working with the `my-ImageNet_Utils` repository: we've downaloded raw ImageNet images there and prepared 250 clean images per wild mushroom category - all stored under `~/repos/my-ImageNet_Utils/shrooms-clean-250-each/`.

We now need to copy the images accordingly to the `my-tensorflow-for-poets`, so we can run some scripts to perform transfer learning and predictions. If you have come across the tensorflow for poets exercise previously, you would have learnt that all the (untracked) working files are stored under `~/repos/my-tensorflow-for-poets/tf_files`. This will include images for retraining, the retrained models / graphs, labels, etc. We can basically store anything relating to retraining in this location for convenience.

#### Prepare 200 Images per category for transfer learning

Create the `shrooms-train-200-each` folder at `~repos/my-tensorflow-for-poets`, and create a similar directory structure to the `shrooms-clean-250-each` that we created and populated earlier:

```
(py27p13) $ cd ~/repos/my-tensorflow-for-poets/tf_files
(py27p13) $ mkdir shrooms-train-200-each
(py27p13) $ cd shrooms-train-200-each
(py27p13) $ mkdir n13003061-fly-agaric
(py27p13) $ mkdir n13030337-scarlet-elf-cup
(py27p13) $ mkdir n13040629-common-stinkhorn
(py27p13) $ mkdir n13044375-giant-puffball
(py27p13) $ mkdir n13044778-Earthstar
```

Now, open up two Finder windows:

- Finder 1 points to the clean images directory: `~/repos/my-ImageNet_Utils/shrooms-clean-250-each`
- Finder 2 points to the retraining images directory: `~/repos/my-tensorflow-for-poets/tf_files/shrooms-train-200-each` 

In Finder 1, sort the images by name. Copy the **first 200** clean images per category over to Finder 2, like this:

<div class="table-wrapper" markdown="block">

| Category          | copy from Finder 1                 | paste to Finder 2               |
|-------------------|------------------------------------|---------------------------------|
| Fly Agaric        | `./n13003061-fly-agaric/`          | `./n13003061-fly-agaric/`       |
| Scarlet Elf cup   | `./n13030337-scarlet-elf-cup/`     | `./n13030337-scarlet-elf-cup/`  |
| Common Stinkhorn  | `./n13040629-common-stinkhorn/`    | `./n13040629-common-stinkhorn/` |
| Giant Puffball    | `./n13044375-giant-puffball/`      | `./n13044375-giant-puffball/`   |
| Earthstar         | `./n13003061-earthstar/`           | `./n13003061-earthstar/`        |

</div>

#### Prepare 50 Images per category for prediction

This will be very similar to our previous step, but we copy over the remaining 50 unused images over for demo / prediction activities later.

Create the `shrooms-demo-50-each` folder at `~repos/my-tensorflow-for-poets`, and create a similar directory structure to the `shrooms-clean-250-each` that we created and populated earlier:

```
(py27p13) $ cd ~/repos/my-tensorflow-for-poets/tf_files
(py27p13) $ mkdir shrooms-demo-50-each
(py27p13) $ cd shrooms-demo-50-each
(py27p13) $ mkdir n13003061-fly-agaric
(py27p13) $ mkdir n13030337-scarlet-elf-cup
(py27p13) $ mkdir n13040629-common-stinkhorn
(py27p13) $ mkdir n13044375-giant-puffball
(py27p13) $ mkdir n13044778-Earthstar
```

Now, open up two Finder windows:

- Finder 1 points to the clean images directory: `~/repos/my-ImageNet_Utils/shrooms-clean-250-each`
- Finder 2 points to the retraining images directory: `~/repos/my-tensorflow-for-poets/tf_files/shrooms-demo-50-each` 

In Finder 1, sort the images by name. Copy the **last 50** clean images per category over to this folder, like this:

<div class="table-wrapper" markdown="block">

| Category          | copy from Finder 1                 | paste to Finder 2               |
|-------------------|------------------------------------|---------------------------------|
| Fly Agaric        | `./n13003061-fly-agaric/`          | `./n13003061-fly-agaric/`       |
| Scarlet Elf cup   | `./n13030337-scarlet-elf-cup/`     | `./n13030337-scarlet-elf-cup/`  |
| Common Stinkhorn  | `./n13040629-common-stinkhorn/`    | `./n13040629-common-stinkhorn/` |
| Giant Puffball    | `./n13044375-giant-puffball/`      | `./n13044375-giant-puffball/`   |
| Earthstar         | `./n13003061-earthstar/`           | `./n13003061-earthstar/`        |

</div>

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
