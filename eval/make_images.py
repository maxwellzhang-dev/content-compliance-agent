"""Draw the synthetic advert mock-ups used by the multimodal eval cases.

They are plain posters with text and simple shapes, generated so the repo ships no
third-party imagery. Run: uv run python eval/make_images.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "images"
W, H = 800, 1000


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("Arial Bold.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def poster(name: str, bg: str, lines: list[tuple[str, int, str]], draw_extra=None) -> None:
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    if draw_extra:
        draw_extra(d)
    y = 80
    for text, size, colour in lines:
        f = font(size)
        for row in text.split("\n"):
            w = d.textlength(row, font=f)
            d.text(((W - w) / 2, y), row, font=f, fill=colour)
            y += int(size * 1.25)
        y += 18
    img.save(OUT / f"{name}.png")


def cartoon_faces(d: ImageDraw.ImageDraw) -> None:
    for cx, colour in [(220, "#ffd166"), (580, "#06d6a0")]:
        d.ellipse([cx - 110, 560, cx + 110, 780], fill=colour, outline="black", width=6)
        d.ellipse([cx - 50, 620, cx - 20, 660], fill="black")
        d.ellipse([cx + 20, 620, cx + 50, 660], fill="black")
        d.arc([cx - 60, 650, cx + 60, 740], 20, 160, fill="black", width=6)


def bottle(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([370, 520, 430, 560], fill="#5b3a1a")
    d.rounded_rectangle([340, 560, 460, 860], radius=30, fill="#8b5a2b")


def phone_post(d: ImageDraw.ImageDraw) -> None:
    d.rounded_rectangle([150, 40, 650, 960], radius=40, outline="#333", width=8, fill="white")
    d.ellipse([190, 90, 260, 160], fill="#f4a3b5")
    d.rounded_rectangle([200, 400, 600, 760], radius=20, fill="#fde2e4")
    d.rectangle([370, 460, 430, 700], fill="#c9184a")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    poster(
        "beer_cartoon",
        "#4cc9f0",
        [
            ("SATURDAY CARTOON FUN!", 54, "white"),
            ("Brought to you by", 34, "#1b1b1b"),
            ("BRAND X BEER", 72, "#7a1f00"),
            ("Airs between your favourite\nmorning cartoons", 32, "#1b1b1b"),
        ],
        lambda d: (cartoon_faces(d), bottle(d)),
    )
    poster(
        "slimming_tea",
        "#f1faee",
        [
            ("SlimTea", 90, "#2a9d8f"),
            ("LOSE 10KG IN 7 DAYS!", 58, "#e63946"),
            ("No diet. No exercise.\nJust one cup a day.", 40, "#1d3557"),
        ],
    )
    poster(
        "ai_travel_labelled",
        "#023047",
        [
            ("Discover Pulau Sentosa", 52, "white"),
            ("Book your weekend escape", 34, "#ffb703"),
            ("AI-generated image", 28, "white"),
        ],
        lambda d: (
            d.ellipse([250, 420, 550, 720], fill="#ffb703"),
            d.rectangle([0, 760, W, H], fill="#219ebc"),
        ),
    )
    poster(
        "crypto_returns",
        "#0b0b0b",
        [
            ("CRYPTOMAX", 80, "#f7931a"),
            ("GUARANTEED 20%\nMONTHLY RETURNS", 60, "white"),
            ("Download the app and start trading today", 30, "#bbbbbb"),
        ],
    )
    poster(
        "unit_trust_compliant",
        "#ffffff",
        [
            ("Lion Bank Global Equity Fund", 44, "#003049"),
            ("Invest for the long term", 36, "#335c67"),
            (
                "Offered by Lion Bank Ltd, a licensed bank.\n"
                "The value of investments can fall as well as rise.\n"
                "Past performance is not a guide to future returns.",
                26,
                "#333333",
            ),
        ],
    )
    poster(
        "influencer_post",
        "#eeeeee",
        [
            ("@glowwithmei", 36, "#222222"),
            ("My honest review of GlowSkin serum", 30, "#222222"),
            ("#skincare #glowup", 28, "#555555"),
        ],
        phone_post,
    )


if __name__ == "__main__":
    main()
