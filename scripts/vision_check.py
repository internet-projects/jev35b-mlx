"""Two generated images through decide(): a choice and a yes/no question each.  python scripts/vision_check.py <model>"""
import sys

from PIL import Image, ImageDraw

from jev_mlx import Jev


def picture(draw):
    im = Image.new("RGB", (448, 448), "white"); draw(ImageDraw.Draw(im)); return im


jev = Jev(sys.argv[1])
options = ["a red circle", "a blue square", "a green triangle", "a yellow star"]
ok = 0
for name, im, want in [("red circle", picture(lambda d: d.ellipse((90, 90, 360, 360), fill="red")), 0),
                       ("blue square", picture(lambda d: d.rectangle((90, 90, 360, 360), fill="blue")), 1)]:
    o = jev.decide("choice", ["Picture:", im], "What does the picture show?", options)
    ok += o["choice_index"] == want
    print(f"{name}: {o['choice']} ({o['probabilities'][o['choice_index']]:.3f})")
    o = jev.decide("noul", [im], "Is the shape in the picture red?")
    ok += (o["choice_index"] == 1) == (want == 0)
    print(f"{name}, red? {o['choice']} ({o['probabilities'][1]:.3f})")
print(f"vision check {ok}/4")
