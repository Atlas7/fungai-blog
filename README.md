Home page for fungai.org. Currently hosted on this GitHub page:

[https://atlas7.github.io/fungai-blog/](https://atlas7.github.io/fungai-blog/)

---

Built with Jekyll [Massively Theme](https://github.com/iwiedenm/jekyll-theme-massively-src), GitHub Pages, [formspree.io](https://formspree.io/), and lots of modifications (e.g. pagination with jekyll-paginate-v2, newer CSS, overall layout, refactoring).

## Development Instruction

git clone the repository. Then `cd` into it.

```
git clone https://github.com/Atlas7/fungai-blog
cd fungai-blog
```

Make sure git repository has the appropriate remote:

```
johnny@Chuns-MBP fungai-blog (master) $ git remote -v
origin  https://github.com/Atlas7/fungai-blog (fetch)
origin  https://github.com/Atlas7/fungai-blog (push)
```

Install dependencies (Ruby gems)

```
bundle install
```

Serve locally:

```
bundle exec jekyll serve
```

This will serve the blog post locally at [http://127.0.0.1:4000/](http://127.0.0.1:4000/)

## How to create a new blog post

Use an IDE (I use Webstorm, though Sublime text will also do) and a terminal (e.g. iterm2, or Mac Terminal).

Create a new blog post:

- Got to `/_posts`
- Create a markdown file with a name like this: `yyyy-mm-dd-hello-this-is-my-title.md`
- Create a meta block at the top (so called "front matters"). I usually just copy and paste from older posts and tweak.
- Write the post in markdown syntax
- Save post

To preview post:

```
bundle exec jekyll serve
```

This will serve the blog post locally at [http://127.0.0.1:4000/](http://127.0.0.1:4000/)

Ready to push to [https://atlas7.github.io/fungai-blog](https://atlas7.github.io/fungai-blog)?

Easy, just do push master to origin:

```
git push origin master
```

(or just `git push`). The actual GitHub page [https://atlas7.github.io/fungai-blog/] may take a few seconds to build / update.

Tip: It is good practice to work on a branch instead of master directly. I usually do this:

```
git checkout -b my-new-branch-name
# do some stuff bla bla bla
git add .
git commit -m "add new features"
git push origin my-new-branch-name
```

Go to Github, create a new pull request. (and as an admin myself I will happily approve and merge). Delete the
GitHub branch afterwards for tidiness (it can always be restored so no worries)

Now that GitHub is updated, update local master too:

```
git pull origin master
```

(or just `git pull`)

Delete the local branch for tidiness:

```
git branch -d my-new-branch-name
```

## Credits

This project will not be as painless / enjoyable if it's not for the Jekyll Massively Theme - which is open source
on GitHub under a Creative Common Licence. Many thanks to these folks!!! (thank you thank you thank you).

### Formspring.io Integration

Formspring is supported out of the box! Just add your email to ```_config.yml```

### Original README from HTML5 UP

```
Massively by HTML5 UP
html5up.net | @ajlkn
Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)


This is Massively, a text-heavy, article-oriented design built around a huge background
image (with a new parallax implementation I'm testing) and scroll effects (powered by
Scrollex). A *slight* departure from all the one-pagers I've been doing lately, but one
that fulfills a few user requests and makes use of some new techniques I've been wanting
to try out. Enjoy it :)

Demo images* courtesy of Unsplash, a radtastic collection of CC0 (public domain) images
you can use for pretty much whatever.

(* = not included)

AJ
aj@lkn.io | @ajlkn


Credits:

	Demo Images:
		Unsplash (unsplash.com)

	Icons:
		Font Awesome (fortawesome.github.com/Font-Awesome)

	Other:
		jQuery (jquery.com)
		Misc. Sass functions (@HugoGiraudel)
		Skel (skel.io)
		Scrollex (github.com/ajlkn/jquery.scrollex)
```
