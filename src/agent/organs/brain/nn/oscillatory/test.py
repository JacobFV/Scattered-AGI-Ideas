import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend

from utils import exponential_periods
from oscillator import Oscillator
from freq_analysis import ConvFreqAnalysis


# this isn't a fair ML challange since this architecture
# is more suited for natural sequences like music or genetic code
B, T = 4, 50
X = tf.random.uniform((B,T,10))
Y = tf.random.uniform((B,T,4))


periods = exponential_periods(2, 5)

integrator = Oscillator(units=10, periods=periods)
differentiator = ConvFreqAnalysis(periods=periods, window_size=12)

model0 = keras.Sequential([
    tfkl.Input(X.shape),
    integrator,
    differentiator,
    tfkl.Dense(4, 'relu')
])
model1 = keras.Sequential([
    tfkl.Input(X.shape),
    integrator,
    tfkl.Dense(35),
    differentiator,
    tfkl.Dense(4, 'relu')
])

model0.compile('Adam', 'mse', ['accuracy'])
model1.compile('Adam', 'mse', ['accuracy'])
model0.fit(X, Y)
model1.fit(X, Y)