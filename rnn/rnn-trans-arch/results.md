# Results
<!---
TODO: 1. Summarize the results of the models in a table and write a short summary of the main conclusions.
2. Add examples of the model predictions.
-->
## Conceptual Questions

### 1. How will the architecture of Fig.1 perform for long sequences? Why?

It will not be able to learn as for two reasons:

1. exploding/vanishing gradients;
2. the encoder architecture uses a fixed-length vector to represent the input sequence. For a long sequence, the decoder will therefore not have sufficient information to represent the input sequence (in other words, there is information loss). Furthermore, the input character is too far away;

### 2. What are some of the techniques we can use to improve the performance of the architecture of Fig.1 for long sequences?

1. Reversing the input sequence (see [Learning to Execute](https://arxiv.org/abs/1410.4615));
2. Clipping the gradients;

### 3. What problems may arise when training with teacher forcing? Consider the differences when we switch from training to testing.

Teacher forcing creates an association between the input ground-truth token and the output. This can be a problem if the training
dataset is not representative. For example, suppose that the word “fox” is always preceded during training by the word “brown”.
The model would learn this correlation between the words, and during generation, it could output “fox”, even though the overall context was the color of a living room (e.g, "I love the brown wall with the painting").

### 4. Can you think of a way to address the issue? Read the paper [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099) for some ideas.

Introduce an interpolation between using the actual token (as in training above) and the generated token (as in generalization).

### GRU vs Linear Attention vs Transformer

The performance of all models is summarized in the table below. We define accuracy as xxx and the similarity score as yyy.

Main conclusions:

1. (GRU);
2. (Linear Attention);
3. (Transformer).
