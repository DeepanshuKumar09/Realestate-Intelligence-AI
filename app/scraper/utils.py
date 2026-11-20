
import json
import re
import time
import heapq

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RealEstateResearchBot/1.0)"
}

REQUEST_DELAY = 1.0

# Pages likely to contain useful real-estate information
PRIORITY_KEYWORDS = [
    "project", "property", "properties", "development",
    "about", "residential", "commercial", "location",
    "service", "community", "investment", "listing"
]

IGNORED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".webp", ".pdf", ".zip", ".css", ".js",
    ".mp4", ".woff", ".woff2"
)


class WebsiteScraper:

    def __init__(self, name, base_url, max_pages=25):

        self.name = name
        self.base_url = base_url.rstrip("/") + "/"
        self.domain = urlparse(self.base_url).netloc
        self.max_pages = max_pages

        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        self.visited = set()
        self.documents = []

        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(
            urljoin(self.base_url, "robots.txt")
        )

        self.robots_allowed = True
        self.sitemaps = []

        self._load_robots()

    def _load_robots(self):

        robots_url = urljoin(self.base_url, "robots.txt")

        try:
            response = self.session.get(
                robots_url,
                timeout=15
            )

            if response.status_code == 200:

                self.robot_parser.parse(
                    response.text.splitlines()
                )

                self.sitemaps = re.findall(
                    r"(?im)^sitemap:\s*(\S+)",
                    response.text
                )

            elif response.status_code in (401, 403):
                self.robots_allowed = False

            # A missing robots.txt means no published rules
            elif response.status_code == 404:
                self.robot_parser.parse([])

            else:
                print(
                    f"Could not verify robots.txt: "
                    f"{response.status_code}"
                )
                self.robots_allowed = False

        except requests.RequestException as error:
            print("Robots check failed:", error)
            self.robots_allowed = False

    def is_valid_url(self, url):

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False

        if parsed.netloc.lower() != self.domain.lower():
            return False

        if parsed.query:
            return False

        if parsed.path.lower().endswith(IGNORED_EXTENSIONS):
            return False

        return True

    def normalize_url(self, url):

        url = urldefrag(url)[0]
        parsed = urlparse(url)

        return parsed._replace(
            path=parsed.path.rstrip("/") + "/",
            query="",
            fragment=""
        ).geturl()

    def allowed_by_robots(self, url):

        return self.robot_parser.can_fetch(
            HEADERS["User-Agent"],
            url
        )

    def get_sitemap_urls(self):

        urls = set()
        pending = list(self.sitemaps)

        # Also check the conventional sitemap location
        pending.append(
            urljoin(self.base_url, "sitemap.xml")
        )

        checked = set()

        while pending and len(checked) < 10:

            sitemap_url = pending.pop(0)

            if sitemap_url in checked:
                continue

            checked.add(sitemap_url)

            try:
                response = self.session.get(
                    sitemap_url,
                    timeout=15
                )

                if response.status_code != 200:
                    continue

                root = ElementTree.fromstring(
                    response.content
                )

                for element in root.iter():

                    if element.tag.endswith("loc"):

                        location = (
                            element.text or ""
                        ).strip()

                        if not location:
                            continue

                        if location.endswith(".xml"):
                            pending.append(location)

                        elif self.is_valid_url(location):
                            urls.add(
                                self.normalize_url(location)
                            )

            except Exception as error:
                print(
                    f"Sitemap error ({sitemap_url}): {error}"
                )

        return urls

    def priority(self, url):

        path = urlparse(url).path.lower()

        score = 0

        for keyword in PRIORITY_KEYWORDS:
            if keyword in path:
                score += 2

        return score

    def extract_page(self, url):

        if not self.allowed_by_robots(url):
            print("Disallowed by robots.txt:", url)
            return None, []

        try:
            response = self.session.get(
                url,
                timeout=20
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            )

            if "text/html" not in content_type.lower():
                return None, []

            soup = BeautifulSoup(
                response.text,
                "lxml"
            )

            title = (
                soup.title.get_text(strip=True)
                if soup.title else ""
            )

            # Remove non-content elements
            for tag in soup([
                "script", "style", "noscript",
                "svg", "nav", "footer", "form"
            ]):
                tag.decompose()

            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.body
                or soup
            )

            headings = [
                tag.get_text(" ", strip=True)
                for tag in main_content.find_all(
                    ["h1", "h2", "h3"]
                )
                if tag.get_text(strip=True)
            ]

            paragraphs = [
                tag.get_text(" ", strip=True)
                for tag in main_content.find_all(
                    ["p", "li"]
                )
                if len(tag.get_text(strip=True)) > 30
            ]

            content = "\n".join(
                headings + paragraphs
            )

            # Discover internal links
            links = set()

            for anchor in soup.find_all(
                "a", href=True
            ):

                link = self.normalize_url(
                    urljoin(url, anchor["href"])
                )

                if self.is_valid_url(link):
                    links.add(link)

            if len(content.strip()) < 100:
                return None, list(links)

            document = {
                "source": self.name,
                "url": url,
                "title": title,
                "headings": headings,
                "content": content,
                "scraped_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }

            return document, list(links)

        except requests.RequestException as error:

            print(f"Page error: {url} | {error}")
            return None, []

    def scrape(self, output_path):

        if not self.robots_allowed:
            print(
                f"Cannot verify scraping permissions for "
                f"{self.name}. Skipping."
            )
            return

        print(f"\nStarting {self.name} scraper...")

        sitemap_urls = self.get_sitemap_urls()

        candidates = sitemap_urls | {
            self.normalize_url(self.base_url)
        }

        # Prioritize relevant pages
        queue = []

        for url in candidates:
            heapq.heappush(
                queue,
                (-self.priority(url), url)
            )

        while queue and len(self.visited) < self.max_pages:

            _, url = heapq.heappop(queue)

            if url in self.visited:
                continue

            self.visited.add(url)

            print(
                f"[{len(self.visited)}/{self.max_pages}] {url}"
            )

            document, links = self.extract_page(url)

            if document:
                self.documents.append(document)

                print(
                    "  Collected:",
                    document["title"]
                )

            # Add newly discovered internal pages
            for link in links:

                if (
                    link not in self.visited
                    and self.is_valid_url(link)
                ):

                    heapq.heappush(
                        queue,
                        (-self.priority(link), link)
                    )

            time.sleep(REQUEST_DELAY)

        self.save_data(output_path)

    def save_data(self, output_path):

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.documents,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("\nScraping completed!")
        print("Website:", self.name)
        print("Pages visited:", len(self.visited))
        print("Documents collected:", len(self.documents))
        print("Saved to:", output_path)
