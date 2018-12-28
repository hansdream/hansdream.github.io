---
layout: default
permalink: /tags/
title: Tags
---


<div class="post">
  <!-- Wrapper Start -->
  <section id="intro" style="border: 1px dotted #ddd;">
    <div class="tags-title">
      <h1>{{ page.title }}</h1>
    </div>
    <br>
    <div class="tag-cloud">
    {% for tag in site.tags %}
      <span style="font-size: {{ tag | last | size | times: 100 | divided_by: site.tags.size | plus: 70  }}%">
        <a href="#{{ tag | first | slugize }}">
          {{ tag | first }}
        </a>
      </span>
    {% endfor %}
    </div>
    <br>

    <div id="archives">
    {% for tag in site.tags %}
      <div class="archive-group">
        <div class="archive-title">
          {% capture tag_name %}{{ tag | first }}{% endcapture %}
          <a name="{{ tag_name | slugize }}"></a>
          <h3 id="#{{ tag_name | slugize }}" class="bold">{{ tag_name }}</h3>
        </div>
        {% for post in site.tags[tag_name] %}
        <article class="archive-item">
          <h4><a class="archive-item-title" href="{{ root_url }}{{ post.url }}">{{post.title}}</a></h4>
        </article>
        {% endfor %}
      </div>

    {% endfor %}
    </div>
  </section>
</div>
