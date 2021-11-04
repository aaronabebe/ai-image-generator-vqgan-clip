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

### install pip dependencies

```sh
pip install -r requirements.txt
```

## running 

Currently you have to change the args in `image_generator.py` manually. 
Simply add a prompt of your liking to try it. 
First visible results can be seen after ~50 iterations. 

```python
python image-generator.py
```


### TODOs

- [ ] setup docker image
- [ ] cli interface for entering string
- [ ] automatically generate gif from progress images
