import pandas as pd
import random
import math

def sigmoid(x):
    
    return 1/(1 + math.exp(-x))
#sigmoid türev
def dsigmoid(y):
    #return sigmoid(x) * (1-sigmoid(x))
    return y * (1-y)


class NeuralNetwork:
    
   def __init__(self, input_nodes, hidden_nodes,output_nodes):
       self.input_nodes = input_nodes
       self.hidden_nodes = hidden_nodes
       self.output_nodes = output_nodes
       
       self.weight_ih = Matrix(self.hidden_nodes,self.input_nodes)
       self.weight_ho = Matrix(self.output_nodes,self.hidden_nodes)
       self.weight_ih.randomize()
       self.weight_ho.randomize()
       
       self.bias_h = Matrix(self.hidden_nodes, 1)
       self.bias_o = Matrix(self.output_nodes, 1)
       self.bias_h.randomize()
       self.bias_o.randomize()
       
       self.learning_rate = 0.5
       
   def FeedForward(self,input_array):
       
       # Giriş verilerini matris haline getir
       inputs = Matrix.fromArray(input_array)
       
       # hidden output
       hidden = Matrix.multiply_matrix(self.weight_ih, inputs)
       hidden.add(self.bias_h)
       #activation function
       hidden.map(sigmoid)
       
       output = Matrix.multiply_matrix(self.weight_ho,hidden )# hidden = hidden output
       output.add(self.bias_o)
       output.map(sigmoid)
       
       return output
   
   def train(self,inputs_array,target_array):
       
       
       inputs = Matrix.fromArray(inputs_array)
       
       hidden = Matrix.multiply_matrix(self.weight_ih, inputs)
       hidden.add(self.bias_h)
       #activation function
       hidden.map(sigmoid)
       
       
       outputs = Matrix.multiply_matrix(self.weight_ho,hidden ) # hidden = hidden output
       outputs.add(self.bias_o)
       outputs.map(sigmoid)
       
       
       targets = Matrix.fromArray(target_array)
       # error = targets - outputs
       output_errors = Matrix.subtract(targets,outputs)
       
       # gradient hesaplanıyor
       gradient = Matrix.map_matrix(outputs,dsigmoid)
       gradient.multiply(output_errors)
       gradient.multiply(self.learning_rate)
        
       hidden_T = Matrix.transpose(hidden)
       weight_ho_deltas = Matrix.multiply_matrix(gradient,hidden_T)
       self.weight_ho.add(weight_ho_deltas)
       self.bias_o.add(gradient)
       
       #hidden error hesaplaması
       who_tranpose = Matrix.transpose(self.weight_ho)
       hidden_errors = Matrix.multiply_matrix(who_tranpose,output_errors)
       
       # hidden gradient hesaplaması
       hidden_gradient = Matrix.map_matrix(hidden, dsigmoid)
       hidden_gradient.multiply(hidden_errors)
       hidden_gradient.multiply(self.learning_rate)
       
       # input hidden delta hesaplaması
       inputs_T = Matrix.transpose(inputs)
       weight_ih_deltas = Matrix.multiply_matrix(hidden_gradient, inputs_T)
       self.weight_ih.add(weight_ih_deltas)
       self.bias_h.add(hidden_gradient)
       
   
class Matrix:
    
    def __init__(self,rows,cols):
        self.rows = rows
        self.cols = cols
        self.data = []
        
        for i in range(self.rows):
            self.data.append([])  # Her satır için boş bir liste ekleyin
            for j in range(self.cols):
                 self.data[i].append(0)  # Her satırın içine sütun sayısı kadar 0 ekleme
          
    def print_matrix(self):
        print(pd.DataFrame(self.data))
    
            
    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = (random.random()*2-1)
    
    
    def add(self, n): 
        if isinstance(n, Matrix): # eğer n bir matrixse matrix toplamı yaparız
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]
                    
        else:
            # Burada n ile matrisin her elemanını toplarız
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n
    @staticmethod           
    def multiply_matrix(a,b):
         if a.cols != b.rows:
             print(" A nin sütünuyla B nin satirlari esit olmali")
             return None
            
         result = Matrix(a.rows,b.cols)
         for i in range(result.rows):
             for j in range(result.cols):
                 sum = 0
                 for k in range (a.cols):
                     sum += a.data[i][k] * b.data[k][j]
                     result.data[i][j]= sum
         return result
                
    def multiply(self, n): 
        # Burada n bir sayı ise matrisin her elemanını çarparız
        if isinstance(n, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n
    @staticmethod        
    def transpose(matrix):
        result = Matrix(matrix.cols,matrix.rows)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result.data[j][i] = matrix.data[i][j]       
        return result
    
    @staticmethod 
    def fromArray(arr):
        m = Matrix(len(arr),1)
        for i in range (len(arr)):
            m.data[i][0]= arr[i]
        return m
    
    
    @staticmethod 
    def subtract(a,b):
        result= Matrix(a.rows, a.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = a.data[i][j] - b.data[i][j]
        
        return result
        
    def toArray(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j]) 
        return arr
    def map(self,func): 
        # Eğer bir matrisle ilgili fonksiyon yazmışsanız ve bu fonksiyonu kullanmak istiyorsanız
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j] # her elemanı value atıyoruz
                self.data[i][j] = func(val)# her eleman için fonksiyonu tekrarlıyor ve matrisin içine atıyor
   
    @staticmethod
    def map_matrix(matrix,func): 
        result= Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                val = matrix.data[i][j] # her elemanı value atıyoruz
                result.data[i][j] = func(val)      
        return result
    
    
    
    
training_data = [
    {"inputs": [0, 1],
     "targets":[1],
     },
    
    {"inputs": [1, 0],
     "targets":[1],
     },
    
    {"inputs": [0, 0],
     "targets":[1],
     },
    
    {"inputs": [0, 0],
     "targets":[0],
     },
    
    {"inputs": [1, 1],
     "targets":[0],
     },
    
]
nn = NeuralNetwork(2,4,1)
for i in range(70000):
    data = random.choice(training_data)
    nn.train(data["inputs"], data["targets"])

result_1 = nn.FeedForward([0, 1]).toArray()
result_2 = nn.FeedForward([1,0]).toArray()
result_4 = nn.FeedForward([1, 1]).toArray()
result_3 = nn.FeedForward([0, 0]).toArray()


df = pd.DataFrame([result_1, result_2,result_4, result_3 ])

print(df)










