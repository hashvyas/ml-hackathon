"""Screenshot the Folium HTML maps in results/maps/ for the PDF (headless Chromium via Playwright).

python report/screenshot_maps.py  -> report/figures/folium_<map name>.png
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1200, "height": 850})
    for html in sorted(Path("results/maps").glob("*.html")):
        page.goto(html.resolve().as_uri())
        page.wait_for_load_state("networkidle")  # wait for map tiles
        page.wait_for_timeout(1500)
        out = Path("report/figures") / f"folium_{html.stem}.png"
        page.screenshot(path=str(out))
        print("wrote", out)
        if "hmm" in html.stem:  # second view without the GPS layer, so the three routes are visible
            page.get_by_text("GPS points", exact=False).last.click()
            page.wait_for_timeout(500)
            out = Path("report/figures") / f"folium_{html.stem}_routes.png"
            page.screenshot(path=str(out))
            print("wrote", out)
    browser.close()
