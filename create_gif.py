from PIL import Image
import glob


DIR = 'example_peter_4'
images = [Image.open(pth) for pth in glob.glob(f'{DIR}/*.png')]

images[0].save(
    f'./{DIR}_gif.gif',
    save_all=True,
    append_images=images[1:],
    duration=100,
    loop=0
)

