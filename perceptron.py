import random
import matplotlib.pyplot as plt


# activitaion function
def sign(n):
    if(n>=0):
        return 1
    else:
        return-1
        
        
class Perceptron:
    
    weights = [0.0,0.0]
    lr = 0.1
    def __init__(self):
        # Ağırlıklar rastgele olarak -1 ile 1 arasında atanıyor
        for i in range(len(self.weights)):
            self.weights[i] = random.uniform(-1, 1)
    
    def guess(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += inputs[i] * self.weights[i]
        output = sign(sum)
        return output
    
    def train(self,inputs,target):
        guess = self.guess(inputs)
        error = target - guess
        # updates all the weights
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.lr
    

class Point:
    def __init__(self):
        self.x = random.uniform(0, 10)
        self.y = random.uniform(0, 10)
        if(self.x > self.y):
            self.label = 1
        else:
            self.label = -1
            
    def show(self):
        # Noktalar grafiğe çizilir, etiketlerine göre renkleri belirlenir
        if self.label == 1: 
            plt.plot(self.x, self.y, marker='o', color='black', markersize=12)  # Label 1 ise siyah renkte büyük bir nokta çiz
        else:
            plt.plot(self.x, self.y, marker='o', color='blue', markersize=12)  # Diğer durumlarda beyaz renkte büyük bir nokta çiz

    
# 100 adet rastgele nokta oluştur
points = [Point() for _ in range(100)]
pc = Perceptron()
for i in range (len(points)):
  points[i] = Point()
# # Her bir nokta grafige çiz
for p in points:
    p.show()
for p in points:
    for p in points:
        inputs = [p.x, p.y]
        target = p.label
        pc.train(inputs, target)
        #guess = pc.guess(inputs)
        # tahminlerin doğruluğuna göre noktalar yeşil veya kırmızı olarak çiz
        if(guess == target):
            plt.plot(p.x,p.y,marker='o', color='green', markersize=5)
        else:
            plt.plot(p.x,p.y, marker='o', color='red', markersize=5)
# line
plt.plot([0, 10], [0, 10], linestyle='dashed', color='gray')

plt.show()







