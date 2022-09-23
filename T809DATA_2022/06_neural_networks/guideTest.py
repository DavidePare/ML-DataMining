import torch
afrom torchvision.models import resnet18, ResNet18_Weights

import numpy as np

'''
Tensors (specialized data structure) are similar to arrays and matrices.
 -> used to encode inputs and outputs of a model
 -> similar to numpy's ndarrays, except that tensors can run on GPUs 


'''

#Tensors can be created drectly from data

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print("First example=",x_data)

#Tensors can be created from numpy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("First example=",x_np)

#From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


#With random or constant value:
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#Tensor attributes
print(f"Shape of tensor: {rand_tensor.shape}")
print(f"Datatype of tensor: {rand_tensor.dtype}")
print(f"Device tensor is stored on: {rand_tensor.device}")


#A lot of operation could be done by CPU
tensor = torch.rand(3, 4)
'''
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
'''

#Is possible concatenate one or more tensor, of same size
t1=torch.cat([rand_tensor,ones_tensor,zeros_tensor],0) # 0 verticale, 1 orizzontale
print(t1)
#Another instruction is stack that concatenates a sequence of tensors (all with the same size) along a new dimension.



#Sarebbe tutti gli elementi al quadrato -> ogni elemento nella matrice moltiplicato per se stesso
# This computes the element-wise product
print(tensor)
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")


#Moltiplicazione tra matrici algebra lineare
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

#In-place operations -> operations that have a _ suffix are in-place. Example x.copy_(t) and x.copy_(y)
print("In-place operations:")
print(tensor, "\n")
tensor.add_(5)
print(tensor)
#Use discouraged, this operations save some memory, but can be problematic when computing derivates because of an immediate loss of history

print("The operation is reflected from the nupy to torch and viceversa")
n = np.ones(5)
t= torch.from_numpy(n) # c'è l operazione che da numpy ti va a torch
np.add(n, 6, out=n)
print(f"t: {t}")
print(f"n: {n}")

# Forward propagation dati dei pesi si va in avanti calcolando un output
# Backward propagation, NN adjusts its parameters proportionate to the error in its guess.

# Esempio
# Diamo un occhiata a un singolo passo di allenamento. Ad esempio carichiamo un pretrained resnet18 modello da
# torchvision e creiamo un tensor di dati randomici per rappresentare una singola immagine con 3 canali
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64) # 3 canali con altezza e larghezza 64
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass -> fare la predizione per vedere il risultato
#Backprop
loss = (prediction - labels).sum() # Calcolare l errore dalla predizione rispetto al corretto risultato
loss.backward() # backward pass, backpropagando quest errore su tutta la network

# Now we load an optimizer SGD (Stochastic gradient descent with) with a learning rate and momentum (https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
#Finally, we call .step() to initiate gradient descent. The optimizer adjusts each parameter by its gradient stored in .grad.
optim.step() #gradient descent


