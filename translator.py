import cv2 
import numpy as np
from PIL import ImageFont, ImageDraw, Image

img = np.zeros((200,1200,3),np.uint8)

fontpath = "C/Users/wysha/Downloads/Tiro_Devanagari_Hindi/"
text = "இனிய பிறந்தநாள் வாழ்த்துக்கள் யோஷினி குட்டி"

font = ImageFont.truetype(fontpath, 32)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((50, 80),text, font = font)
img_tamil = np.array(img_pil)

cv2.imwrite('tamil.jpg', img_tamil)
