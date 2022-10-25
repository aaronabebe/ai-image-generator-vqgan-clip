# ai-image-generator

- Original idea written By Katherine Crowson (https://github.com/crowsonkb,
  https://twitter.com/RiversHaveWings).
- The original BigGAN+CLIP method was by https://twitter.com/advadnoun.

## setup

### clone dependencies from git

```sh
git clone https://github.com/openai/CLIP
git clone https://github.com/CompVis/taming-transformers
```

### Download model weights

Download the model weights from
here: [Link to instructions](https://github.com/EleutherAI/vqgan-clip/blob/main/README.md)

### install pip dependencies

```sh
pip install -r requirements.txt
```

## running

Provide a csv file in the given format, same as in `test_instructions.csv`.
Then run the generation program like this to generate the frames for an animation:

```sh
python cli.py --instruction_path 'test_instructions.csv'
```

## TODOs

### feature ideas 

- [x] automatically generate gif from progress images
- [x] generate animation frames from csv file
- [x] continue frame generation for folder/csv/framenumber
- [ ] add some form of automated ai upscaling, e.g. [with opencv](https://learnopencv.com/super-resolution-in-opencv/)
- [ ] add optional inversion/passe-partout similar to breitband video idea
- [ ] add functionality to let it run on videos (e.g. get frames/prompts -> run 50 iterations for each frame)
- [ ] cli interface for entering string
- [ ] setup docker image
- [ ] 
- [ ] 
