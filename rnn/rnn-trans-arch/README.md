# Encoder-Decoder RNN models for sequence-to-sequence learning

The encoder-decoder architecture is a neural network design pattern used in sequence-to-sequence learning. It is composed of two recurrent neural networks (RNNs): an encoder and a decoder. The encoder processes the input sequence, compressing it into a fixed-size internal representation. The decoder reads this internal representation and generates the output sequence, by producing a distribution over the output vocabulary conditioned on the previous hidden state and the output token in the previous time step.

This example is mostly derived from [assignment 3](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/assignments/assignment3.pdf) of the Neural Networks and Deep Learning course of the University of Toronto. See more information in the course page [here](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/).

The goal is to translate Pig Latin to English. Pig Latin is a language game that involves altering English words according to a simple set of rules. For example, the word "pig" is translated to "igpay" and the word "latin" is translated to "atinlay". The model is trained on a dataset of Pig Latin words and their English translations.

## Results Summary

For more details about the architecture, check the `architecture.md` file.
