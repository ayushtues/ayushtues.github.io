"""Convert Medium posts into Hugo page bundles under content/blog/.

Usage:
  python medium2hugo.py rss    <feed.xml>         <site_dir>
  python medium2hugo.py export <medium-export-dir> <site_dir>

Existing post folders are never overwritten, so re-running is safe.
"""
import email.utils
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

UA = {"User-Agent": "Mozilla/5.0"}


class Converter(MarkdownConverter):
    def __init__(self, images, **kw):
        super().__init__(heading_style="ATX", bullets="-", **kw)
        self.images = images  # remote url -> local filename

    def convert_figure(self, el, text, *args, **kw):
        img = el.find("img")
        cap = el.find("figcaption")
        if img is None:  # embeds (videos, gifs, gists): shortcode or link
            a = el.find("a") or el.find("iframe")
            src = a and (a.get("href") or a.get("src"))
            if not src:
                return "\n\n"
            yt = re.search(r"youtube\.com/embed/([\w-]+)", src)
            cap_md = ""
            if cap and cap.get_text(strip=True):
                cap_md = "\n\n*" + self.process_tag(cap, {"_inline"}).strip() + "*"
            if yt:
                return f"\n\n{{{{< youtube {yt.group(1)} >}}}}{cap_md}\n\n"
            return f"\n\n[Embedded content]({src}){cap_md}\n\n"
        src = self.images.get(img.get("src"), img.get("src"))
        alt = (img.get("alt") or "").replace('"', "'")
        if cap and cap.get_text(strip=True):
            c = self.process_tag(cap, {"_inline"}).strip().replace('"', '\\"')
            return f'\n\n{{{{< figure src="{src}" alt="{alt}" caption="{c}" >}}}}\n\n'
        return f"\n\n![{alt}]({src})\n\n"

    def convert_img(self, el, text, *args, **kw):
        src = self.images.get(el.get("src"), el.get("src"))
        return f"![{el.get('alt') or ''}]({src})"

    def convert_pre(self, el, text, *args, **kw):
        for br in el.find_all("br"):
            br.replace_with("\n")
        code = el.get_text().strip("\n")
        lang = el.get("data-code-block-lang") or ""
        return f"\n\n```{lang}\n{code}\n```\n\n"


def slugify(url):
    last = url.split("?")[0].rstrip("/").split("/")[-1]
    return re.sub(r"-[0-9a-f]{8,}$", "", last) or last


def yaml_str(s):
    return json.dumps(s, ensure_ascii=False)


def download_images(soup, out_dir):
    for frame in soup.find_all("iframe"):
        g = re.search(r"giphy\.com/embed/(\w+)", frame.get("src", ""))
        if g:
            frame.replace_with(soup.new_tag("img", src=f"https://media.giphy.com/media/{g.group(1)}/giphy.gif", alt="GIF"))
    for img in soup.find_all("img"):
        if img.get("src"):
            img["src"] = img["src"].replace("cdn-images-1.medium.com/max/800/", "cdn-images-1.medium.com/max/1600/")
    images, n = {}, 0
    for img in soup.find_all("img"):
        src = img.get("src") or ""
        if "/_/stat?" in src:
            img.decompose()
            continue
        if not src.startswith("http") or src in images:
            continue
        n += 1
        ext = os.path.splitext(src.split("?")[0])[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            ext = ".jpg"
        name = f"image-{n}{ext}"
        for attempt in range(6):
            try:
                fetch = re.sub(r"https://cdn-images-1\.medium\.com/max/\d+/", "https://miro.medium.com/v2/resize:fit:1600/", src)
                with urllib.request.urlopen(urllib.request.Request(fetch, headers=UA), timeout=60) as r:
                    open(os.path.join(out_dir, name), "wb").write(r.read())
                images[src] = name
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 5:
                    print(f"  ! could not download {src}: {e}")
                    break
                time.sleep(10 * 2**attempt)
            except Exception as e:
                print(f"  ! could not download {src}: {e}")
                break
        time.sleep(0.5)
    return images


def write_post(site, title, summary, date, tags, url, body_html):
    slug = slugify(url)
    out = os.path.join(site, "content", "blog", slug)
    if os.path.exists(out):
        print(f"skip (exists): {slug}")
        return
    os.makedirs(out)
    soup = BeautifulSoup(body_html, "html.parser")

    # Medium repeats the title (and subtitle) as the first heading(s) of the body.
    for _ in range(2):
        first = next((c for c in soup.contents if getattr(c, "name", None)), None)
        if first is None or first.name not in ("h1", "h3", "h4"):
            break
        t = " ".join(first.get_text(" ", strip=True).split())
        if t == " ".join(title.split()):
            first.decompose()
        elif not summary:
            summary = t
            first.decompose()
        else:
            break

    images = download_images(soup, out)
    md = Converter(images).convert_soup(soup)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    fm = ["---", f"title: {yaml_str(title)}"]
    if summary:
        fm.append(f"summary: {yaml_str(summary)}")
    fm.append(f"date: {date.strftime('%Y-%m-%d')}")
    fm += ["authors:", "  - admin"]
    if tags:
        fm.append("tags:")
        fm += [f"  - {yaml_str(t)}" for t in tags]
    if images:
        # Hugo Blox uses the first image as the card thumbnail / header.
        first_img = next(iter(images.values()))
        ext = os.path.splitext(first_img)[1]
        os.link(os.path.join(out, first_img), os.path.join(out, "featured" + ext))
        fm += ["image:", "  caption: ''", "  preview_only: true"]
    fm += [f"# Originally published on Medium: {url}", "---", ""]
    footer = f"\n\n---\n\n*Originally published on [Medium]({url}).*\n"
    open(os.path.join(out, "index.md"), "w").write("\n".join(fm) + "\n" + md + footer)
    print(f"wrote: {slug} ({len(images)} images)")


def tag_name(t):
    return t.replace("-", " ").title()


def from_rss(path, site):
    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
    for item in ET.parse(path).getroot().iter("item"):
        url = item.findtext("link").split("?")[0]
        write_post(
            site,
            title=item.findtext("title"),
            summary="",
            date=email.utils.parsedate_to_datetime(item.findtext("pubDate")),
            tags=[tag_name(c.text) for c in item.findall("category")],
            url=url,
            body_html=item.find("content:encoded", ns).text,
        )


def from_export(export_dir, site):
    posts = os.path.join(export_dir, "posts")
    for fn in sorted(os.listdir(posts)):
        if not fn.endswith(".html") or fn.startswith("draft_"):
            continue
        soup = BeautifulSoup(open(os.path.join(posts, fn)).read(), "html.parser")
        if not soup.find(class_="graf--title"):
            print(f"skip (reply to another post): {fn}")
            continue
        canon = soup.find("a", class_="p-canonical")
        published = soup.find("time", class_="dt-published")
        body = soup.find("section", attrs={"data-field": "body"})
        if not (canon and published and body):
            print(f"skip (not a published post): {fn}")
            continue
        sub = soup.find("section", attrs={"data-field": "subtitle"})
        # Unwrap Medium's section/div layout wrappers so headings sit at top level.
        for el in body.select(".graf--title, .graf--subtitle"):
            el.decompose()
        inner = BeautifulSoup("", "html.parser")
        for el in body.select(".section-inner > *"):
            inner.append(el)
        write_post(
            site,
            title=soup.find("h1", class_="p-name").get_text(strip=True),
            summary=sub.get_text(" ", strip=True) if sub else "",
            date=datetime.fromisoformat(published["datetime"].replace("Z", "+00:00")),
            tags=[],
            url=canon["href"],
            body_html=str(inner),
        )


if __name__ == "__main__":
    mode, src, site = sys.argv[1:4]
    {"rss": from_rss, "export": from_export}[mode](src, site)
