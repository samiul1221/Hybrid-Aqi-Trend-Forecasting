# renders the draw.io SVG to a vector PDF (+ a 3x PNG preview) with the installed Edge, so Fig. 1 stays sharp in print
from pathlib import Path
from playwright.sync_api import sync_playwright
here = Path(__file__).parent / "figures"
svg = (here / "fig1_architecture.svg").read_text(encoding="utf8")
page_html = "<html><body style='margin:0'>" + svg.split("?>", 1)[-1].replace('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">', "") + "</body></html>"
(here / "_render.html").write_text(page_html, encoding="utf8")
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    pg = b.new_page(viewport={"width": 1172, "height": 687}, device_scale_factor=3)
    pg.goto((here / "_render.html").resolve().as_uri())
    pg.wait_for_timeout(500)
    pg.screenshot(path=str(here / "fig1_architecture_3x.png"))
    pg.pdf(path=str(here / "fig1_architecture.pdf"), width="1172px", height="687px", print_background=True, page_ranges="1")
    b.close()
