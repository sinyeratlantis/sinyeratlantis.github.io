## DeepMind PySC2 星际争霸2强化学习环境搭建文档

**我们使用python3.6作为我们的开发语言。**

### Windows

- 如果未曾使用过暴雪战网客户端，下载战网桌面客户端（https://www.battlenet.com.cn/download/getInstallerForGame?os=win&locale=zhCN&version=LIVE&gameProgram=BATTLENET_APP）并执行安装。

  安装完成后，在第一次打开战网时，选择星际争霸2并安装，若无战网账号请先按照流程注册。

  若使用过战网客户端，在左上角暴雪标志处打开下拉菜单，选择设置，在弹出的页面左侧选择“游戏安装/更新”，在星际争霸2一栏输入Starcraft2文件夹路径。

  定位完毕后在战网主页右侧启动栏选择星际争霸2，跳过教程，启动一次正式游戏，画面中央是动画，上方是标题栏，右下角是聊天界面，即为加载成功。

- 在Windows中安装好python环境，不会问百度，然后用pip安装PySC2环境

  ```shell
  pip install pysc2
  ```

  习惯使用conda的同学也可以在conda环境中安装，没有问题。

- 设置系统环境变量

  以win7为例，右键计算机，选择属性，右侧选择高级系统设置，在tab栏选择高级，在右下角选择环境变量，打开后找到下方的系统变量，点击新建。

  变量名设置为（注意大写）：SC2PATH

  路径设置为你所拷贝的Starcraft2文件夹路径。

- 在命令行执行测试

  ```shell
  python -m pysc2.bin.agent --map Simple64
  ```

- 在IDE中执行测试

  推荐使用pycharm，然后配置好python的解释器。

  首先测试 ```import pysc2```

  若无报错，则访问deepmind的agent.py文件：https://github.com/deepmind/pysc2/blob/master/pysc2/bin/agent.py，将其中的代码拷贝到IDE中，将第75，76行代码

  ```python
  flags.DEFINE_string("map", None, "Name of a map to use.")
  flags.mark_flag_as_required("map")
  ```

  替换为

  ```python
  flags.DEFINE_string("map", "Simple64", "Name of a map to use.")
  ```

  若成功启动星际争霸2调试环境，则环境搭建成功。

### Mac

- 下载战网桌面客户端（https://www.battlenet.com.cn/download/getInstallerForGame?os=mac&locale=zhCN&version=LIVE&gameProgram=BATTLENET_APP）并安装。

  安装完成后，在第一次打开战网时，选择星际争霸2并安装，若无战网账号请先按照流程注册。

  若使用过战网客户端，在左上角暴雪标志处打开下拉菜单，选择设置，在弹出的页面左侧选择“游戏安装/更新”，在星际争霸2一栏输入Starcraft2文件夹路径。

  定位完毕后在战网主页右侧启动栏选择星际争霸2，跳过教程，启动一次正式游戏，画面中央是动画，上方是标题栏，右下角是聊天界面，即为加载成功。

- 用pip安装PySC2环境

  ```shell
  pip install pysc2
  ```

  习惯使用conda的同学也可以在conda环境中安装，没有问题。

- 在命令行执行测试

  ```shell
  python -m pysc2.bin.agent --map Simple64
  ```

- 在IDE中执行测试

  同Windows

执行成功会出现如下界面

![](pic/pysc2_install/1.png)

![](pic/pysc2_install/2.png)

![](pic/pysc2_install/3.png)
