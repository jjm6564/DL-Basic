import matplotlib.pyplot as plt

#1) draw a line
plt.title('test')
plt.plot([1, 2, 3,4], [2, 4, 8,6])

#2) axis-labels
plt.xlabel('hours')
plt.ylabel('scores')

#3) add line / legends
plt.plot([1.5, 2.5, 3.5,4.5], [3, 5, 8,10])
plt.legend(['A students','B students'])

plt.show()



