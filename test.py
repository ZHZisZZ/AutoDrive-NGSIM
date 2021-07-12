import numpy as np

def wrapper():
  cnt = 0
  def func():
    nonlocal cnt
    cnt += 1
    print(cnt)
  return func

func = wrapper()
func()
func()

