---
layout: post
title:  "IDZ Interview"
date:   2017-12-06 12:30:00
excerpt: "Intel Developer Zone Interview (draft)"
image: https://avatars1.githubusercontent.com/u/11135428?s=400&u=6974cfe92abcde2c79bcf492b31cb9908c5c1818&v=4
---

#### Tell us about your background.

Professionally I've worked as a technologist for a large global investment bank, an analytics consultant for a major UK commercial and private bank, and a full-stack developer for a wellness startup. In my spare time I am the creator and author of [Mathalope.co.uk](http://mathalope.co.uk/) (a tech blog visited by 120,000+ students and professionals from 180+ countries), volunteer developer of the [Friends of Russia Dock Woodland Website](http://fordw.org/), open source software contributor, hackathon competitor, and part of the Intel Software Innovator Program. I am working my way on becomming a better machine learning engineer and educator.

#### What got you started in technology?

Whilst I was studying for my aeronautical engineering masters at Imperial College London, I learnt to write small Fortran/Matlab programs where you can throw at it say satellite data, and it spit out your geographical location on Earth. I then started my professional technology career in 2008 for a global investment bank, where I collaborated with colleagues from all 4 regions globally (EMEA, ASPAC, NAM and LATAM), developed and rolled-out a fully automated data extraction and analytics tool - protecting the 100,000+ production systems (Windows, UNIX, AIX, Mainframe platforms etc) from risk of overloading. I built the system with proprietary technologies such as SAS, SQL, Oracle, Autosys batch scheuling, and internal configuration databases. In 2014 I decided to learn about open source technologies in my spare time and as a result created [Mathalope.co.uk](http://mathalope.co.uk/) - and that's the point I believe was where I started making a real impact globally - contributing codes and knowledge to the rest of the world, reaching to 120,000+ audiences from 180+ countries.

#### What projects are you working on now?

I am currently building [fungAI.org](http://fungai.org/) - a machine learning application that automatically identify wild mushroom species from input images using deep learning (classification and localization) techniques. The project was primarily motivated by a casual friend's Facebook post from a walking trip:
 
 > "hey do you know what mushroom this is?"
  
Coincidentally [my partner who is a conservationist](https://twitter.com/lemon_disco) happens also to be a mushroom enthusiast and so naturally we've formed a [couple team](http://127.0.0.1:4000/team/) - me focus on the tech part and Clare our "chief mushroom domain expert adviser". We think the project would potentially be fun and educational.

You can [read more about the project concept here](http://fungai.org/concept/), take a look at an [initial frontend toy demo here](https://fungai-react-ui.herokuapp.com/fungpredict), and check out this [Intel DevMesh Fungi Barbarian Project page](https://devmesh.intel.com/projects/fungi-barbarian). All source codes to the projects are [open sourced on GitHub](http://fungai.org/demos/).

![fungi-barbarian-concept-v2.png](/images/blog/fungi-barbarian-concept-v2.png)

![fungai-poc-1.png](/images/blog/fungai-poc-1.png)

![fungai-poc-2.png](/images/blog/fungai-poc-2.png)

![fungai-poc-3.png](/images/blog/fungai-poc-3.png)

#### Tell us about a technology challenge you’ve had to overcome in a project.

During the summer of 2015 I spent the entire weekend just trying to get [OpenCV-Python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html) installed on my Windows PC via the [Anaconda package manager](https://docs.anaconda.com/anaconda/). I recall "google-ing" for solutions, trying out many of them, and failed. After many rounds of trial-and-error and investigations I solve the problem by combining multiple "partially working" solutions that I found on the internet. In the end I decided to [write an article summarising my solution via a blog post](http://mathalope.co.uk/2015/05/07/opencv-python-how-to-install-opencv-python-package-to-anaconda-windows/) which has been viewed more than 120,000 times since! Just to increase the range of impact I also immediately [posted it as a solution to a Stackoverflow Forum](https://stackoverflow.com/questions/23119413/how-do-i-install-python-opencv-through-conda#answer-30281466) and obtained 50+ upvotes for it from the community (turned out that forum was viewed over 200,000+ times!). The other thing was those "hey thank you it worked" types of feedback comments - it turns out many developers around the world had similar issues, tried out my solution, and eventually solved the problem at their end.

This experience has taught me the importance of making an impact - it doesn't have to be building the next Google or Facebook. It can be as simple as writing up a summary of how you've solved a problem and sharing it online. We only get to live once.
This experience has taught me the importance of making an impact - it doesn't have to be building the next Google or Facebook. It can be as simple as writing up a summary of how you've solved a problem and sharing it online. We only get to live once.
This experience has taught me the importance of making an impact - it doesn't have to be building the next Google or Facebook. It can be as simple as writing up a summary of how you've solved a problem and sharing it online. We only get to live once.

![opencv-conda-blog-comment-4.png](/images/blog/opencv-conda-blog-comment-4.png)

#### What trends do you see happening in technology in the near future?

A recent talk published by O'Reilly in September 2017: [AI is the New Electricity by Andrew Ng](https://www.youtube.com/watch?v=NQK4ZY_gwKI) discussed the trends and value creation of machine learning. This is a one-liner summary:

> Today, vast majority of values across industry is created by Supervised Learning, and closely followed by Transfer Learning

Personally, I am super excited about transfer learning and believe this technique will be used everywhere by everybody - including mere mortals like me for [fungAI.org](http://fungai.org/)!

Say we wish to [train a model to recognise different types of daisies](https://www.tensorflow.org/tutorials/image_retraining) for instance. Instead of having to spend months training a model from scratch with millions of images, we can take a (very massive) short-cut: take a pre-trained model like [Inception v3](https://www.kaggle.com/google-brain/inception-v3) that is already very good at regconising objects from [ImageNet data](http://www.image-net.org/). Use it as a starting point and train that more specialised daisy recognition model from there. The end result? You only require about 200 images per daisy category, and the training of a new model would take only about 30 minutes on a modern laptop on CPU.

> This suddenly makes deep learning very inclusive to everybody
 
An ultra powerful and expensive GPU is no longer a "must have requirement" to solve deep learning problems. Transfer learning and open source software together have made deep learning more inclusive and accessible to all. This will lead to stronger communities, knowledge sharing, and further technological advancement in the near future.

#### How does Intel help you succeed?

Intel has supported innovative projects, such as [fungAI.org](http://fungai.org) (which I'm working on currently) by providing access to state-of-the-art deep learning technologies, such as their Intel Xeon-Phi enabled Cluster Nodes for model training, and Neural Compute Stick for embedded machine learning applications. At a personal level, the Intel has provided me access to a community of technology experts and innovators ranging from AI, IoT, VR and Game Development - where I get to learn and bounce ideas from. I really appreciate the amount of efforts that the coordinators of the Intel Software Innovator Program has put in to ensure success of the community.

#### Outside of technology, what type of hobbies do you enjoy?

- touch / tag ruby
- piano
- cycling
