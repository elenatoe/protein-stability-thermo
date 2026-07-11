# protein-stability-thermo

Python implementation of thermodynamic models for protein stability prediction,
built from biophysical chemistry coursework.

## Overview

- **CHC model** (`chc_model.py`) — fits denaturation curves to extract
  thermodynamic stability parameters.
- **Helix-coil transfer matrix** (`helix_coil.py`) — predicts helix
  propensity along a sequence using the transfer matrix method.

Validated against synthetic datasets generated to test method correctness;
no unpublished lab or coursework data is included in this repository.

## Status

In development. Core CHC fitting implemented; helix-coil module in progress.

## Install

\`\`\`bash
git clone https://github.com/elenatoe/protein-stability-thermo.git
cd protein-stability-thermo
pip install -r requirements.txt
\`\`\`

## License

MIT
