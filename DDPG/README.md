# Deep Deterministic Policy Gradient

DDPG is an off-policy algortihm designed for environments with continuous action spaces. It has two distinct networks, a critic which learns the action values by minimizing the TD error, and an actor which learns the policy. The critic side of DDPG is similar to process in DQN and the same tricks (replay buffers and target networks) are also applied here. However, the actor in DDPG is novel but fairly simple. The action values are differentiable with respect to actions, thus a simple gradient ascent (- gradient descent) can be utilized to update the actors parameters.

$$
\min_{\theta} -\mathbb{E}_{s \sim \mathcal{D}}[ s, \mu_{\theta}(s)]
$$

where $\mu_{\theta}$ is the critic.

Also, a random noise is added to the actions during training phase for the sake of exploration.

![](gifs/pendulum.gif) |  
:-------------------------:|
The result of trained DDPG agent after 500 episodes for Pendulum environment.