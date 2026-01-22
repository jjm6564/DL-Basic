import pandas as pd

sr = pd.Series([1, 2, 3, 4],
               index=['a', 'b', 'c', 'd'])

print('-'*15)
print(sr)

print('seirses value : {}'.format(sr.values))
print('series index : {}'.format(sr.index))

values = [[1,2,3], [4,5,6], [7,8,9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)
print('#'*18)
print('데이터프레임 출력 : ')
print('데이터프레임 인덱스 : {}'.format(df.index))
print('데이터프레임 컬럼 : {}'.format(df.columns))
print('-'*18)
print(df)
print('-'*18)
print(df.values)

#리스트 생성
data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]

df = pd.DataFrame(data)
df = pd.DataFrame(data,columns=['학번','이름','점수'])
print(df)

# 딕셔너리로 생성하기
data = {
    '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
    '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
    '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
    }

df = pd.DataFrame(data)
print(df)
