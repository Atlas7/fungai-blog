---
layout: post
title:  "IDZ Interview"
date:   2017-12-06 12:30:00
excerpt: "Intel Developer Zone Interview (draft)"
image: https://avatars1.githubusercontent.com/u/11135428?s=400&u=6974cfe92abcde2c79bcf492b31cb9908c5c1818&v=4
---

#### Tell us about your background.

Professionally I've been a technologist for a large global investment bank, an analytics consultant for a major UK commercial and private bank, a full-stack developer for a wellness startup, and briefly an engineering intern for an airline. In my spare time I am the creator and author of [Mathalope.co.uk](http://mathalope.co.uk/) (a tech blog visited by 120,000+ students and professionals from 180+ countries), volunteer developer of the [Friends of Russia Dock Woodland Website](http://fordw.org/), open source software contributor, hackathon competitor, and part of the Intel Software Innovator Program. I am working my way on becomming a better machine learning engineer and educator.

#### What got you started in technology?

When I was studying for my masters in aeronautical engineering at Imperial College London, I learnt to write small Fortran/Matlab programs where you can throw at it say satellite data, and it spit out geographical location on Earth. I then started my professional technology career in 2008 for a global investment bank, where I collaborated with colleagues from all 4 regions globally (EMEA, ASPAC, NAM and LATAM), developed and rolled-out a fully automated data extraction and analytics tool - protecting the 100,000+ production systems (Windows, UNIX, AIX, Mainframe platforms etc) from risk of overloading. I built the system with proprietary technologies including SAS, SQL, Oracle, Autosys batch scheuling, and internal configuration databases. In 2014 I decided to learn about open source technologies in my spare time and as a result created [Mathalope.co.uk](http://mathalope.co.uk/) - and that's the point I believe was where I started making a real impact globally - contributing codes and knowledge to the rest of the world, reaching to 120,000+ audiences from 180+ countries.

#### What projects are you working on now?

I am currently building [fungAI.org](http://fungai.org/) - a machine learning application with the aim of identifying wild mushroom species from images using deep learning (classification and localization) techniques. The project was primarily inspired and motivated by a casual friend's Facebook post from a walking trip:
 
 > "hey do you know what mushroom this is?"
  
Coincidentally [my partner who is a conservationist](https://twitter.com/lemon_disco) happens also to be a mushroom enthusiast and so naturally we've formed a [couple team](http://127.0.0.1:4000/team/) - me focus on the tech part and Clare our "chief mushroom domain expert adviser". We think the project will be fun and educational.

You can [read more about the project concept](http://fungai.org/concept/), try out an [initial frontend toy demo](https://fungai-react-ui.herokuapp.com/fungpredict), and check out this [Intel DevMesh Fungi Barbarian Project page](https://devmesh.intel.com/projects/fungi-barbarian). All project source codes are open sourced on GitHub - you may find [more Demos and Github repository links here](http://fungai.org/demos/).

![fungi-barbarian-concept-v2.png](/images/blog/fungi-barbarian-concept-v2.png)

![fungai-poc-1.png](/images/blog/fungai-poc-1.png)

![fungai-poc-2.png](/images/blog/fungai-poc-2.png)

#### Tell us about a technology challenge you’ve had to overcome in a project.

During the summer of 2015 I spent the entire weekend just trying to get [OpenCV-Python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html), Windows, and the [Anaconda package manager](https://docs.anaconda.com/anaconda/) to work together for a personal computer vision project. I remember searching really hard on the internet for solutions, trying out many of them, and failed uncountable times. After many rounds of trial-and-errors and investigations I eventually solved the problem by combining multiple "partially working" solutions that I found on the internet. In the end I decided to [write an article summarising my solution via a blog post](http://mathalope.co.uk/2015/05/07/opencv-python-how-to-install-opencv-python-package-to-anaconda-windows/) which has since been viewed more than 120,000 times. To increase the range of impact I also [posted it as a solution to a Stackoverflow Forum](https://stackoverflow.com/questions/23119413/how-do-i-install-python-opencv-through-conda#answer-30281466) - the forum has so far been viewed over 200,000 times and my solution has received 50+ "good citizen brownie points" upvotes. To my surprise, the posts have received significant "hey thank you it worked" types of feedback comments. It turns out many developers around the world had also bumped into similar issues at the time and got the problem solved with the help of the articles.

This experience has taught me the importance of making an impact: it doesn't have to be building the next Google or Facebook. Instead, it can be as simple as writing up a summary of how you've solved a problem and sharing it online. We only get to live once.

![opencv-conda-blog-comment-4.png](/images/blog/opencv-conda-blog-comment-4.png)

#### What trends do you see happening in technology in the near future?

A recent talk published by O'Reilly in September 2017: [AI is the New Electricity by Andrew Ng](https://www.youtube.com/watch?v=NQK4ZY_gwKI) discussed the trends and value creation of machine learning. This is my a one-liner summary take from Andrew:

> Today, vast majority of values across industry is created by Supervised Learning, and closely followed by Transfer Learning

Personally, I am super excited about transfer learning and believe this technique will be used everywhere by everybody - including mere mortals like me for [fungAI.org](http://fungai.org/).

Say we wish to [train a model to recognise different types of daisies](https://www.tensorflow.org/tutorials/image_retraining) for instance. Instead of having to spend months training a model from scratch with millions of images, we can take a very massive short-cut: take a pre-trained model like [Inception v3](https://www.kaggle.com/google-brain/inception-v3) that is already very good at regconising objects from [ImageNet data](http://www.image-net.org/), use it as a starting point and train that more specialised daisy recognition model from there. The end result? You only require about 200 images per daisy category, and the training of a new model would take only about 30 minutes on a modern laptop on CPU.

> This suddenly makes deep learning very inclusive to everybody
 
An ultra powerful and expensive GPU is no longer a "must have requirement" to solve deep learning problems. Transfer learning and open source software together have made deep learning more inclusive and accessible to all. This will lead to stronger communities, knowledge sharing, and further technological advancement of deep learning in the near future.

![fungai-poc-3.png](/images/blog/fungai-poc-3.png)

#### How does Intel help you succeed?

Intel supports innovative projects (including [fungAI.org](http://fungai.org) which I'm currently working on) by providing access to state-of-the-art deep learning technologies, including the Intel Xeon-Phi enabled cluster nodes for model training, Neural Compute Stick for embedded machine learning applications, and more. At a personal level, Intel has provided me access to a community of technology experts and innovators from AI, IoT, VR and Game Development - where I get to learn and bounce ideas from. I was also sponsored by Intel to take part in events including the Seattle Innovator Summit and Nuremberg Embedded World Expo, where I had the opportunity to travel, learn, and contribute to the tech community. I really appreciate the amount of efforts the Intel Software Innovator Program team has put in to ensure long-term success of the innovation community. It has been a privilege and I thank you all for the opportunities.

#### Outside of technology, what type of hobbies do you enjoy?

Since 2009 I've been playing social mixed-gender non-contact (touch/tag) rugby leagues here in London weekly evenings outside work. It's a fun way to socialise and meet new friends. What's even better: the lost calories from exercising on the field is perfectly balanced by the calories gained from the pub session afterwards. (But to be honest I think it's usually a net gain!)
