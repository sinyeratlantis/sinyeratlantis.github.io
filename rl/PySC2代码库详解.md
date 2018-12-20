## PySC2详解

PySC2源码中包含了几个部分：

- agents：一些基本的agent，包括random agent和针对小任务的script agent
- bin：启动程序，分为agent启动和human启动
- env：rl research的环境接口以及agent与env相互作用的关系
- lib：定义了一些运行时的依赖，如观测量、动作量及特征层的定义
- maps：定义了一些针对地图的设置
- run_configs：定义了一些关于游戏运行的设置
- tests：定义了PySC2的一些单元测试

bin负责定义游戏的运行模式，主要包括通过agent操作和人直接操作两种。通过agent操作对应的是bin.agent模块，人直接操作对应的是bin.play模块。运行PySC2时必须调用这两个模块中的某一个。我们暂时只关心通过agent操作。

### 1. agent模块

agent模块主要完成的功能是根据需求打开多个线程。每个线程中，分别从env中生成星际2的环境，从agents中调用agent模块，并利用env.run_loop模块执行环境和代理之间的循环交互。这种运行模式下，一共有15个命令行参数可以定制，其中部分为增强学习环境的参数。15个参数分别为：

1. render: 是否通过pygame渲染游戏画面。默认为True。
2. screen_resolution: 游戏画面分辨率。默认为84。
3. minimap_resolution: 小地图分辨率。默认为64。
4. max_agent_steps: agent的最大步数。默认为2500。
5. game_steps_per_episode: 每个运行分段的游戏步数。默认为0，表示没有限制。若为None，表示使用地图的默认设置。
6. step_mul: agent的每一步中游戏推进步数。这一参数决定了agent操作的速度，默认为8，约等于180APM，和中等水平人类玩家相当。若为None，表示使用地图的默认设置。
7. agent: 指定运行哪个代理。默认为自带的随机代理pysc2.agents.random_agent.RandomAgent。
8. agent_race: agent的种族。默认为None，即Protoss、Terran、Zerg中随机产生。
9. bot_race: 游戏AI的种族。默认为None，即Protoss、Terran、Zerg中随机产生。
10. difficulty: 游戏难度。默认为None，表示VeryEasy。对于难度设置，有对应关系：1 --> VeryEasy, 2 --> Easy, 3 --> Medium, 4 --> MediumHard, 5 --> Hard, 6 --> Harder, 7 --> VeryHard, 8 --> CheatVision (视野作弊), 9 --> CheatMoney (金钱作弊), A --> CheatInsane (疯狂作弊)。
11. profile: 是否打开代码分析功能。默认为False。
12. trace: 是否追溯代码执行。默认为False。
13. parallel: 并行运行多少个实例（线程）。默认为1。
14. save_replay: 是否在结束时保存游戏回放。默认为True。
15. map: 将要使用的地图名字。默认为None。

### 2. 环境

SC2Env环境实例化时有15个可以设置的参数。不在agent模块中有定义的参数如下：

1. map_name: 地图名称。
2. screen_size_px: 游戏画面大小（像素点）。默认为(64, 64)。
3. minimap_size_px: 小地图大小（像素点）。默认为(64, 64)。
4. camera_width_world_units: 用真实世界单位来衡量的游戏画面宽度（摄像机视角）。例如摄像机视角宽度为24，游戏画面宽度为默认的64个像素，则每个像素代表24 / 64 = 0.375个真实世界单位。默认为None，即24。一般不用考虑。
5. discount: 增强学习中回报的折现系数。默认为1。
6. visualize: 是否显示画面。默认为False。
7. save_replay_steps: 每次保存回放时间隔的游戏步数。默认为0，即不保存。
8. replay_dir: 保存回放的路径。默认为None。
9. score_index: 得分（回报）的指标方式。-1代表采用赢或输的环境回报作为增强学习训练的输入；>=0 意味着选取观测量累计得分score_cumulative中的某一种得分指标，例如所有单位的总价值total_value_units。总共有13种指标可以选择，详见lib.features中score_cumulative的定义。默认为None，即采用地图默认值。
10. score_multiplier: 得分（回报）的放大系数。默认为None，即采用地图默认值。

### 3. 观测量

观测量集合，其实现细节则是在lib.features中Features类的observation_spec方法。观测量主要包括12种：

1. screen: 游戏画面信息。SCREEN_FEATURES包含了13种游戏画面特征，例如地形图height_map和可见地图visibility_map等。详见文档: https://github.com/deepmind/pysc2/blob/master/docs/environment.md。同时，这些特征层也显示在界面右侧。
2. minimap: 小地图信息。MINIMAP_FEATURES包含了7种小地图特征，例如地形图height_map和可见地图visibility_map等。详见文档: https://github.com/deepmind/pysc2/blob/master/docs/environment.md。同时，这些特征层也显示在界面右侧。
3. player: 玩家信息。
4. game_loop: 游戏循环。张量大小为(1,)。
5. score_cumulative: 累计得分。张量大小为(13,)。总共有13种得分信息，包括所有单位的总价值total_value_units等。
6. available_actions: 目前观测情况下的可用动作集合。张量大小为(0,)。0代表可变长度。
7. single_select: 单个被选择的单位的信息。张量大小为(0, 7)。总共有7种单位信息，包括单位类型unit type、生命值health等。
8. multi_select: 多个被选择的单位的信息。张量大小为(0, 7)。总共有7种信息，与single_select一致。
9. cargo: 运输工具中所有单位的信息。张量大小为(0, 7)。总共有7种信息，与single_select一致。与unload动作搭配使用。
10. cargo_slots_available: 运输工具中可用的空位。张量大小为(1,)。
11. build_queue: 一个生产建筑正在生产的所有单位的信息。张量大小为(0, 7)。总共有7种信息，与single_select一致。与build_queue动作搭配使用。
12. control_groups: 控制编组的信息。张量大小为(10, 2)。2种信息分别为10个控制编组中每个编组领头的单位类型和数量。与control-group动作搭配使用。

### 4. 动作函数

总共有524个动作函数，都满足这样的形式

```
<函数ID>/<函数名>(<参数ID>/<参数类型> [<参数值大小>, *]; *)
```

其中，函数ID和函数名是唯一的。这些动作函数又具有不同的函数类型，每个函数类型具有特定的参数类型。因此，每个动作函数都有与之对应的参数。具体的参数类型有13个：

1. screen: 游戏画面中的一个点。 
2. minimap: 小地图中的一个点。 
3. screen2: 一个矩形的第二个点。（定义的函数无法接受两个同类型的参数，因此需要定义screen2。） 
4. queued: 这个动作是否现在执行，还是延迟执行。 
5. control_group_act: 对控制编组做什么。 
6. control_group_id: 对哪个控制编组执行动作。 
7. select_point_act: 对这个点上的单位做什么。 
8. select_add: 是否添加这个单位到已经选择的单位中或者替换它。 
9. select_unit_act: 对通过ID选择的单位做什么。 
10. select_unit_id: 通过ID选择哪个单位。 
11. select_worker: 选择一个生产者做什么。 
12. build_queue_id: 选择哪个生产序列。 
13. unload_id: 从交通工具/虫洞/指挥中心中卸载哪个单位。

### 5. agent设计

agent完成的功能是接收观测量并处理，选择合适的动作，并与环境交互，主要是在agents中定义的。其中，agents.base_agent模块定义了agent的基类BaseAgent，其最主要的方法是step，即接收观测量，并返回动作。在此基础上，PySC2内置了两种演示子类，分别是随机agent（agents.random_agent.RandomAgent）和针对小任务定制的脚本agent（agents.scripted_agent中的3种）。

对于我们自己构造的agent，与这些内置的类似，我们需要继承BaseAgent基类，并重写step方法。由env.run_loop可以看出，timesteps直接给到了agent的step方法作为输入。对timesteps分解，包含以下四个部分：

1. timesteps.step_type: 状态类型（是首末状态或其他）
2. timesteps.reward: 回报
3. timesteps.discount: 折现系数
4. timesteps.observation: 观测量集合，字典类型。对于某个观测量可以通过类似timesteps.observation["available_actions"]的方式获取

一个简单的agent应该具有如下形式

```python
class SimpleAgent(base_agent.BaseAgent):
    """简单代理"""
    def step(self, timesteps):
        super(SimpleAgent, self).step(timesteps)
        #############
        由观测量生成动作
        #############
        return actions.FunctionCall(function_id, args)
```

其中，function_id为动作函数ID，args为动作函数的参数。例如，对于select_point动作，可以有

```python
return lib.actions.FunctionCall(2, [[0], [23, 38]]) 
```

到这里，我们已经可以自己编写一个简单的脚本agent了。

对于强化学习的agent，只有step方法是不够的，还需要有一系列方法使其能够训练。以policy gradient为例，agent至少需要具备以下几个方法：

```python
class PolicyGradientAgent(base_agent.BaseAgent):
    """策略梯度代理"""
    def setup(self, obs_spec, action_spec): # 初始化，覆盖基类setup方法

    def create_policy_net(self): # 构造策略网络

    def create_training_method(self): # 构造训练方法

    def train_policy_net(self): # 训练策略网络

    def step(self, obs): # 根据观测量输出动作，覆盖基类step方法
```

同时，env.run_loop只能用于一个固定的agent与环境的循环交互。在训练过程中，我们还需自己编写训练代码，调用PolicyGradientAgent和SC2Env进行交互。

### 6. policy的设计

动作函数+参数这种描述方式，与一般的动作集合有很大的不同，对policy network（策略网络）的构造造成了很大的困难。一般的动作集合中，一个动作 $a$ 是一个单独的动作。构造策略网络，就是通过一系列网络参数来逼近关于动作 $a$ 的单变量概率分布 $\pi(a|s)$。在PySC2中，一个动作 $a$ 实际上是一个包含了多个单独动作的list，包括了动作函数 $a^0$ 以及一系列参数 $a^l$。所以，构造策略网络，拟合的是关于动作 $a$ 的多变量联合概率分布 $\pi(a|s)$。

比较直观的想法是假定这 $L$ 个随机变量是相互独立的，直接构造网络表示 $\pi(a|s)$，但由于动作函数和参数都较多，尤其是画面坐标这些离散参数范围较大，导致这样的采样空间非常大，有101938719种可能的动作。所以，这需要非常多的网络参数才能刻画这个分布，另一方面，采样空间中无效部分太大，也影响训练效果。

pysc2的论文里提供了另一种思路，核心思想是发掘这 $L$ 个随机变量之间内在的依赖关系，从而剔除不合理的组合，减小采样空间。对于某个特定的动作函数 $a^0$ ，并不一定要求给出全部 $L$ 个参数，每个参数也不要求采样全部的参数类型。例如，no_op（表示不采取任何操作）这个动作函数，完全不需要任何参数，这种情况下认为其他参数是独立的并仍然进行采样，显然是十分低效的。再如，move_screen（将选择的单位移动到画面中某一点）这个动作函数，只需要两个参数，参数类型也是确定的（即queued和screen），同时参数范围也是明确的。利用链式法则，论文中将这种思想通过自回归的方式表示为 $\pi(a|s)$，实际上，是将选择一组完整动作的 $a$ 问题转化为对各个参数 $a^l$ 的序列决策问题。

另外，对于不可用动作的限制，论文是通过对动作函数 $a^0$ 的概率分布重新规范化来实现的。至于具体的算法实现细节及源代码，DeepMind并未公布，可以作为一个研究方向。





参考文章：https://zhuanlan.zhihu.com/p/28434323