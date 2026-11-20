from pathlib import Path
from app.scraper.utils import WebsiteScraper


BASE_DIR = Path(__file__).resolve().parents[2]

URL = "https://wasalt.sa/"

OUTPUT = BASE_DIR / "data" / "raw" / "wasalt.json"


def main():

    scraper = WebsiteScraper(
        name="Wasalt",
        base_url=URL,
        max_pages=25
    )

    scraper.scrape(OUTPUT)


if __name__ == "__main__":
    main()