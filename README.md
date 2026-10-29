# neural-network-from-scratch

A implementation of a neural network based on PyTorch `torch.nn` module style.
Full article explanation is contained in [nn.ipynb](notebooks/nn.ipynb).

<img src="images/demo.png" width="500px" alt="demo"/>

## Getting Started

Clone and install the project from source

```bash
git clone https://github.com/joeyshi12/neural-network-from-scratch.git
cd neural-network-from-scratch
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Testing the model

Test the model by running inference against digits that you draw on a digital canvas.

```bash
digitpaint models/example1.model
```

### Controls

- Clear canvas: c
- Guess digit with model: \<space\>

## Training the model

Code for training the digit recognition neural network can be found in [train.ipynb](notebooks/train.ipynb).

To run the code yourself, install `jupyterlab` with `pip3 install jupyterlab` and run `jupyter lab` to open the web IDE.
