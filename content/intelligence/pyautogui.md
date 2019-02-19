## pyautogui的一些使用

```python

import pyautogui

pyautogui.size()  # 屏幕尺寸

pyautogui.moveTo(100, 100, duration=0.25)  # 将鼠标立即移动到屏幕的指定位置

pyautogui.moveRel(100, 0, duration=0.25)  # 相对于当前的位置移动鼠标

pyautogui.position()  # 返回鼠标当前的位置

pyautogui.click(100, 150, button='left')  # 在相应位置点击鼠标左键

pyautogui.mouseDown()  # 按下鼠标按键

pyautogui.mouseDown()  # 释放鼠标按键

pyautogui.doubleClick()  # 执行双击鼠标左键

pyautogui.rightClick()  # 执行双击右键

pyautogui.middleClick()  # 执行双击中键

pyautogui.dragTo()  # 保持点击拖动鼠标

pyautogui.dragRel()  # 保持点击拖动鼠标

pyautogui.scroll(100)  # 滚动鼠标（mac负数向下）

im = pyautogui.screenshot()  # 获取屏幕快照

im.getpixel((50, 200))  # 获取像素值

pyautogui.typewrite('Hello world!')  # 向计算机发送虚拟按键
# 配合time.sleep可解决禁止粘贴问题

pyautogui.typewrite(['a', 'b', 'left', 'X'])
# a键、b键、左箭头、X键，回车键：'enter'(or 'return' or '\n')，Esc键：'esc'

pyautogui.keyDown('shift')  # 按下

pyautogui.press('4')  # 按键

pyautogui.keyUp('shift')  # 释放

pyautogui.hotkey('ctrl', 'c')  # 热键Ctrl-C


```

