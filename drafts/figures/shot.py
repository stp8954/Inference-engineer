import asyncio, pathlib
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(device_scale_factor=2)
        for f in sorted(pathlib.Path(".").glob("*.svg")):
            svg = f.read_text()
            w = int(svg.split('width="')[1].split('"')[0])
            h = int(svg.split('height="')[1].split('"')[0])
            await pg.set_viewport_size({"width": w, "height": h})
            await pg.set_content(f'<body style="margin:0">{svg}</body>')
            await pg.screenshot(path=str(f.with_suffix(".png")))
            print("rendered", f.with_suffix(".png").name, f"{w}x{h} @2x")
        await b.close()
asyncio.run(main())
