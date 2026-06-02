from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://fbref.com/en/comps/1/schedule/World-Cup-Scores-and-Fixtures", timeout=60000)
        title = page.title()
        print(f"Title: {title}")
        browser.close()

if __name__ == '__main__':
    run()
