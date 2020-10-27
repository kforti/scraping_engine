import json

import pytesseract
from PIL import Image


class Transcriber:
    def __init__(self, name):
        self.name = name

    def transcribe(self):
        raise NotImplemented

    def serialize(self):
        raise NotImplemented

    def evaluate(self, label, prediction):
        raise NotImplemented

    @classmethod
    def deserialize(cls, s):
        data = json.loads(s)
        transcriber = cls(**data)
        return transcriber


class TesseractTranscriber(Transcriber):
    def __init__(self, name, convert_options, to_string_options, **kwargs):
        self.convert_options = convert_options
        self.to_string_options = to_string_options
        super().__init__(name, **kwargs)

    def transcribe(self, path):
        img = Image.open(path)
        img = img.convert(**self.convert_options)
        strs = pytesseract.image_to_string(img, **self.to_string_options)
        return strs

    def serialize(self):
        return json.dumps({'name': self.name,
                           'convert_options': self.convert_options,
                          'to_string_options': self.to_string_options})




def crop_image(im, bbox):
    if isinstance(im, str):
        im = Image.open(im)
    cropim = im.crop(bbox)
    cropim.save("test_crop.png")
    return "test_crop.png"


bbox = {"top":273,"left":850,"height":36,"width": 619} # title
bbox = {"top":1172,"left":7,"height":328,"width":731} # description
path = "/home/kevin/bin/scraping_engine/data/product_images_2/2VVDHY2H9QUD_0.png" # title
path = "/home/kevin/bin/scraping_engine/data/product_images_2/350UZWE91DCJ_1.png" # description
l = bbox['left']
r = bbox['left'] + bbox['width']
t = bbox['top']
b = bbox['top'] + bbox['height']

cp = crop_image(path, (l, t, r, b))
eng_transcriber = TesseractTranscriber("eng", {"mode": "RGB"}, {'lang': 'eng', 'config': "--psm 4 --oem 1"})
s = eng_transcriber.transcribe(cp)
print(s)