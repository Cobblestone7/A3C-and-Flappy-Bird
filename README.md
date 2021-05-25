# Asynchronous Advantage Actor-Critic and Flappy Bird

An A3C reinforcement learning algorithm implementation for Flappy Bird.  
This GitHub repository contains the code for out bachelor's degree project:
Asynchronous Advantage Actor-Critic and Flappy Bird.

### Authors:
Markus Fredriksson, GitHub: https://github.com/MarkusFredriksson  
Marcus Wibrink

## Abstract:
Games provide ideal environments for assessing reinforcement learning algorithms because of their simple dynamics and their inexpensive testing, compared to real-world environments. Asynchronous Advantage Actor-Critic (A3C), developed by DeepMind, has shown significant improvements in performance over other state-of-the-art algorithms on Atari games. Additionally, the algorithm A3C(lambda) which is a generalization of A3C, has previously been shown to further improve upon A3C in these environments. 
In this work, we implement A3C and A3C(lambda) on the environment Cart-Pole and Flappy Bird and evaluate their performance via simulation. The simulations show that A3C effectively masters the Cart-Pole environment, as expected. In Flappy Bird sparse rewards are present, and the simulations reveal that despite this A3C manages to overcome this challenge the majority of times, achieving a linear increase in learning. Further simulations were made on Flappy Bird with the inclusion of an entropy term and with A3C(lambda), which display no signs of improvement in performance when compared to regular A3C. 

