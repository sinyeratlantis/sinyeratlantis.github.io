## pysc2 note

### challenge: 

- multi-agent: many players and each player controls many units

- imperfect information

- action space is vast and diverse

- game last many frames

### contribution: 

- observations and actions are defined by features

- rewards based on score provided by sc2 engine

- mini-games

- human players dataset

![](pic/pysc2_note/1.png)

### release components: 

- linux sc2 binary

- sc2 api: start game, observations, actions, review replay (no reward)

- **pysc2**

  **based on sc2 api, defines an action and observation specification**

  includes a random agent and a handful of rule-based agents as example

  includes some mini-games

### environment detail: 

![](pic/pysc2_note/2.png)

- 1v1, build-in ai, reward in two ways: win/tie/loss or blizzard score

- observation in feature layers, many concepts are provided: unit type, hit points, owner, visibility ...

- top down orthographic projection render feature layers in same size, while game use 3d perspective camera with varible size of view

- pysc2-play is for human to play

- human actions between 30 to 300 per minute, pysc2 act every 8 frames, about 180 actions per minute

  ![](pic/pysc2_note/3.png)

- mini-game, StarCraft Map Editor, pysc2/blob/master/docs/mini_games.md


### baseline agents

details in paper

- a3c

- policy representation

- architecture

  ![](pic/pysc2_note/4.png)

  - atari-net agent: two layer convnet, 16/32 filters, 8/4 size, stride 4/2, linear layer to handle non-spatial features

  - fully convnet agent: screen and minimap use separate 2-layer convnet with 16/32 filters of size 5/3 respectively

  - fcn with lstm

### experiment

- truncate the trajectory and run backpropagation after K = 40 forward steps of a network or if a terminal signal is received
- 64 asynchronous threads using shared RMSProp
- 100 experiments, each using randomly sampled hyper-parameters
- entropy penalty of $10^{-3}$ for the action function and each action-function argument
- All experiments were run for 600M steps (or 8×600M game steps)

