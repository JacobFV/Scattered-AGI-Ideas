# Artificial General Intelligence

Open research to all benevolent agents. Experiments hosted here supplement qualitative observations in my blog [Meditating on AI](https://jacobfv.github.io/blog/). I welcome all informative comments and criticism on my scientific process and encourage you to replicate results yourself.

## How to reproduce an experiment

Most experiments are developed in Google Colab using jupyter notebooks. Please visit [this tutorial](https://colab.research.google.com/notebooks/intro.ipynb) for a quick introduction. NOTE: Some experiments require significant compute and memory which may not be available under the free teir. 

1. Go to [Google Colaboratory](https://colab.research.google.com/)
2. Choose `File` &rarr; `Open notebook` &rarr; Github
3. Enter this Github repository identifier `JacobFV/AGI` in the search field
4. Select the notebook you want to test

## Conventions
test
Most of my notebooks employ the following abbreviations:
```python
#@title imports
%tensorflow_version 2.x

import math
import tqdm
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

!pip install -q tsalib
import tsalib
import networkx
!pip install -q jplotlib
import jplotlib as jpl
!pip install -q livelossplot
from livelossplot import PlotLossesKeras

import tensorflow as tf
keras = tf.keras
tfkl = keras.layers

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
```

## License

All code is published under the [MIT License](https://github.com/JacobFV/AGI/blob/master/LICENSE)

```
MIT License

Copyright (c) 2021 Jacob Valdez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
