# Artificial General Intelligence

Open research to all benevolent agents. Experiments hosted here supplement qualitative observations in my blog [Meditating on AI](https://jacobfv.github.io/blog/). I welcome all informative comments and criticism on my scientific process and encourage you to replicate results yourself.

## How to reproduce an experiment

Most experiments are developed in Google Colab using jupyter notebooks. Please visit [this tutorial](https://colab.research.google.com/notebooks/intro.ipynb) for a quick introduction. NOTE: Some experiments require significant compute and memory which may not be available under the free teir. 

1. Go to [Google Colaboratory](https://colab.research.google.com/)
2. Choose `File` &rarr; `Open notebook` &rarr; Github
3. Enter this Github repository identifier `JacobFV/AGI` in the search field
4. Select the notebook you want to test

## Conventions

Most of my notebooks employ the following abbreviations:
```python
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
