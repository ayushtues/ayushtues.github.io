---
# Leave the homepage title empty to use the site title
title: ""
date: 2022-10-24
type: landing

design:
  spacing: "2.5rem"

sections:
  - block: home-hero
    id: about
    content:
      username: admin

  - block: pub-list
    id: publications
    content:
      title: Publications

  - block: post-list
    id: writing
    content:
      title: Writing
      subtitle: Essays, paper explainers, and the occasional rant.
      count: 5
      more:
        text: All posts
        url: blog/

  - block: card-grid
    id: projects
    content:
      title: Projects
      subtitle: Things I've built, mostly papers implemented from scratch and open-source contributions.
      section: project

  - block: card-grid
    id: videos
    content:
      title: Videos
      subtitle: Paper explainers on my [YouTube channel](https://www.youtube.com/channel/UCGEWHZv8Gn10Lk56PzCeFMA).
      section: videos
      count: 8
    design:
      variant: video
---
