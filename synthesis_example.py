"""
Project: Programmatic Synthesis
File: synthesis_example.py
Author: Tristan Onek

Purpose: This is a very minimal project that enables greater control over the generative art process as compared
to other projects that I've worked on.  In this program, the input source material is left up to the programmer
as opposed to the program itself, but the image distortion and merging process is still done by the
machine.  The specific image distortion effects can be chosen and changed by the programmer, however.

This program uses an iterative process to creating the output.  Additional image distortion effects
from 'imgaug' further manipulate the final product.  Unlike with my other projects, this one affords
the user more control regarding which distortion effects can be used on top of certain images.  Essentially,
this project is a basic experiment in generating aesthetic distortion.

I started this small side project with the specific goal of allowing myself the ability to manually select
the images and distortion effects that I could use to generate new art.
By synthesizing old art in the public domain with modern computational methods, it is possible to give that
art a second chance in the current artistic discussion.
"""

from PIL import Image
from PIL import ImageFilter as ifl
import imgaug.augmenters as iaa
import numpy as np
import os

print('Program started')

isDir = os.path.isdir('scraped_art')
if not isDir:
    print('You need a directory called scraped_art for this project to work, with images in that directory'
          'numerically numbered.  Refer to the file calls below to see what this should look like in practice.')
    exit()

lb = 3
ub = 7
jigrow = 7
jigcol = 7

jigsaw = iaa.Jigsaw(nb_rows=jigrow, nb_cols=jigcol)

rrgv = iaa.RelativeRegularGridVoronoi(
    (0.03, 0.1), 0.1, p_drop_points=0.0, p_replace=0.9, max_size=2048)

emb = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1))

warp = iaa.WithPolarWarping(iaa.AveragePooling((2, 8)))

# Merge the first two images together to get the first new.png then start the loop
background = Image.open("scraped_art/1.jpg")
overlay = Image.open("scraped_art/2.jpg")
background = background.convert("RGBA")
overlay = overlay.convert("RGBA")
new_img = Image.blend(background, overlay, 0.25)
new_img.save("new.png", "PNG")

# This loop assumes 6 images total, start with scraped image #3 that gets synthesized on top of #1 and #2 from above
for i in range(lb, ub):
    background = Image.open("new.png")
    overlay = Image.open("scraped_art/" + str(i) + ".jpg")
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    new_img = Image.blend(background, overlay, 0.25)
    new_img.save("new.png", "PNG")

new_img = Image.blend(background, overlay, 0.5)
new_img = new_img.filter(ifl.ModeFilter(size=15))

new_img.save("new.png","PNG", dpi=(300,300))

image = Image.open('new.png')               #open the image
numpyim = np.array(image)                   #convert to np image type
image_aug = jigsaw(image=numpyim)
im = Image.fromarray(image_aug)
im.save('finalproduct.png', dpi=(300,300))

image = Image.open('finalproduct.png')      #open the image
numpyim = np.array(image)                   #convert to np image type
image_aug = emb(image=numpyim)
im = Image.fromarray(image_aug)
im.save('finalproduct.png', dpi=(300,300))
print('Program finished.')