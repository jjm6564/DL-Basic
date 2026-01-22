import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#zip() : 여러 개의 순회 가능한(iterable) 자료형을 인자로 받아 각 자료형의 동일한 인덱스에 위치한 원소들을 튜플로 묶어주는 함수
#ex) a=[1,2,3], b=[4,5,6]
#zip(a,b) -> [(1,4),(2,5),(3,6)]

X, y = zip(['a',1], ['b',2], ['c',3])
print('X 데이터: ',X)
print('y 데이터: ',y)
print('--'*20)

sequece = [['a',1], ['b',2], ['c',3]]
X, y = zip(*sequece) #*는 언패킹 연산자
print('X 데이터: ',X)
print('y 데이터: ',y)
print('--'*20)