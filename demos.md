---
layout: page
title: Demos
description: Demos
sitemap:
    priority: 0.7
    lastmod: 2017-11-02
    changefreq: weekly
image: "/images/blog/fungai-poc-1.png"
---

## Demos

![page.image]({{page.image}})

Here are some initial proof of concepts (POC). May take a while to spin up.

### Intel Developer Mesh Project

- Done! Created <a href="https://devmesh.intel.com/projects/fungi-barbarian" target="_blank">this DevMesh Project - Fungi Barbarian</a>.
  Use this page and share project updates with rest of Intel Developer Zone community.

### Blog

- Done! It is [this website - fungai.org](http://fungai.org)
    <a href="{{ site.github_repo_url }}" class="icon fa-github" rel="nofollow" target="_blank">
      <span class="label">GitHub</span>
    </a>

### Batch Fungi Classification User Interface (UI)

- <a href="https://fungai-react-ui.herokuapp.com/fungpredict" target="_blank">ReactJS UI built with ReactJS and
  hosted on Heroku</a>
    <a href="https://github.com/Atlas7/fungai-react-ui" class="icon fa-github" rel="nofollow" target="_blank">
      <span class="label">GitHub</span>
    </a>
- <a href="https://fungai-json-server-heroku.herokuapp.com/" target="_blank">Fake API built with json-server and
  hosted Heroku</a>
    <a href="https://github.com/Atlas7/fungai-json-server-heroku" class="icon fa-github" rel="nofollow" target="_blank">
      <span class="label">GitHub</span>
    </a>

### Tensorflow Models - Training

- (coming soon) build a simple tensorflow classification model to recognise 3 catetories: Fly Agaric,
  Common Stinkhorn, and Scarlet Elf-cup. Expecting to use transfer learning on Inception v3 model for this.
- (coming soon) build a better tensorflow classification model to recognise more categories. 
  
### Tensorflow Models - Serving

Once we have a pre-trained Tensorflow model (or versions of models), we may integrate it to a frontend application,
such as web, mobile, and even <a href="https://www.raspberrypi.org/" target="_target">Raspberry Pi</a> with
Intel <a href="https://developer.movidius.com/" target="_blank">Movidius Neural Compute Stick</a>.

- (coming soon) serve pre-trained Tensorflow model to a web app
- (coming soon) serve pre-trained Tensorflow model to a mobile app (iOS, Android)
- (coming soon) serve pre-trained Tensorflow model to Raspberry Pi

Will update this page from time to time...