# MLPvsDQN
I pitted two versions of a multi-layer perceptron network and a deep Q neural network against each other.
Performance is based on their average and highest reward over 100 episodes.
The second MLP requires the DQN to run first, but after the DQN runs it saves the data for the
second MLP in 'data.txt', so after running it the first time you won't have to run it again. I have,
however, left the data file from my DQN run should you wish to just run it.  
TensorFlow version: 1.15  
Gym version: 0.17  
Sources:  
OpenAI Gym  
- Gym: https://github.com/openai/gym
- Cartpole: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py  
TheComputerScientist  
- Source Code: https://github.com/the-computerscientist/OpenAIGym/blob/master/DeepQNetworksInOpenAIGym.ipynb
- In-depth Deep Q Network Tutorial: https://youtu.be/dpBKz1wxE_c
- OpenAI Gym Tutorial: https://youtu.be/8MC3y7ASoPs
