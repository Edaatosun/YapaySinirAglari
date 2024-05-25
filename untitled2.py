import random
import matplotlib.pyplot as plt
import numpy as np

# activitaion function
def sign(n):
    if(n>=0):
        return 1
    else:
        return-1



def f(x):
    # y = mx+b
    return 0.3*x+0.2


width = 10
height = 10  


class Perceptron:
    
    weights = [0.0,0.0]
    # learning rate 
    lr = 0.1
    def __init__(self):
        for i in range(len(self.weights)):
            self.weights[i] = random.uniform(-1, 1)
    
    def guess(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += inputs[i] * self.weights[i] # f(x) = w0.x0 + w1.x1+....
        output = sign(sum)
        return output
    
    def train(self,inputs,target):
        guess = self.guess(inputs)
        error = target - guess
        # updates all the weights
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.lr # new weights = error * x0 * learning reate
    

class Point:
    
    def __init__(self):
        self.x = random.uniform(-1,1)
        self.y = random.uniform(-1,1)
        if(self.x > self.y):
            self.label = 1
        else:
            self.label = -1
            
    def pointss(self,x_,y_):
        x_ = self.x
        y_ = self. y
    def pixelx(self):
        return np.interp(self.x, [-1, 1], [0, width])
    def pixely(self):
        return np.interp(self.y, [-1, 1], [0, height])
    def show(self):
        if self.label == 1: 
            plt.plot(self.x, self.y, marker='o', color='black', markersize=12)  # Label 1 ise siyah renkte büyük bir nokta çiz
        else:
            plt.plot(self.x, self.y, marker='o', color='blue', markersize=12)  # Diğer durumlarda beyaz renkte büyük bir nokta çiz
        px = p.pixelx()
        py = p.pixely()
        circle = plt.Circle((px,py), radius=0.1, color='red', fill=False)
        plt.gca().add_patch(circle)

points = [Point() for _ in range(100)]
pc = Perceptron()
for i in range (len(points)):
  points[i] = Point()
# Her bir noktayı çiz
for p in points:
    p.show()
for p in points:
    for p in points:
        inputs = [p.x, p.y]
        target = p.label
        pc.train(inputs, target)
        guess = pc.guess(inputs)
        if(guess == target):
            plt.plot(p.pixelx(),p.pixely(),marker='o', color='green', markersize=5)
        else:
            plt.plot(p.pixelx(),p.pixely(), marker='o', color='red', markersize=5)
# linee
p1 = Point()
p2 = Point()
p1.x = -1
p1.y = f(-1)
p2.x = 1
p2.y = f(1)

plt.plot([p1.pixelx(), p2.pixelx()], [p1.pixely(), p2.pixely()], linestyle='dashed', color='gray')

# plt.plot([0, width], [height,0], linestyle='dashed', color='gray')
plt.show()













