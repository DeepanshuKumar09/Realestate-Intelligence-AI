from pathlib import Path
from app.scraper.utils import WebsiteScraper


BASE_DIR = Path(__file__).resolve().parents[2]

URL = "https://darglobal.co.uk/"

OUTPUT = BASE_DIR / "data" / "raw" / "darglobal.json"


def main():

    scraper = WebsiteScraper(
        name="DarGlobal",
        base_url=URL,
        max_pages=25
    )

    scraper.scrape(OUTPUT)


if __name__ == "__main__":
    main()