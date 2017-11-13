import numpy as np 

#x is the size of input neuron, and h is the size of output neuron
def weight_init(x, h):
	return np.random.uniform(size = (x, h) )	

def bias_init(h):
	return np.random.uniform(size = (1, h) )

def sigmoid(z):
	if z.any()<0:
		return np.exp(z)/(1+np.exp(z))
	else:
		return 1/(1+np.exp(-z))

def sigmoid_prime(z):
	return z * (1 - z)

def cost(t, y):
	return -np.mean(np.dot(t.T(), np.log(y) ) )

def error(y, t):
	return (y-t) * sigmoid_prime(y)

def hidden_grad(e, h):
	return e* sigmoid_prime(h) 

def out_grad(e, h):
	return np.dot(h.T, e)

def bo_grad(e):
	return np.sum(e, axis=0, keepdims = True)

def bh_grad(e, h):
	return np.sum(e * sigmoid_prime(h), axis=0, keepdims = True)

def accuracy(y, t):
	n = y.shape[0]
	y = [0 if o<0.5 else 1 for o in y]
	print(y, n, t)
	t=t.reshape(-1)
	print(t)
	return np.sum(y==t)/n

def predict(w, x):
	return sigmoid(np.dot(x, w))



#some data
X=np.array([[1, 1],[0, 0],[1, 0], [0, 1]])

#Output
y=np.array([[1],[0],[0], [0]])

#Define params
EPOCH = 500
lr = 0.1
hidden_node1 = 3
hidden_node2 = 3

#define initial weights:
wh1 = weight_init(X.shape[1], hidden_node1)
bh1 = bias_init(hidden_node1)
wh2 = weight_init(wh1.shape[1], hidden_node2 )
bh2 = bias_init(hidden_node2)
wo = weight_init(hidden_node2, 1)	
bo = bias_init(1) 

for i in range(EPOCH):
	#feedforward:

	#first hidden layer
	h_in1 = np.dot(X, wh1) + bh1
	h_out1 = sigmoid(h_in1)

	h_in2 = np.dot(h_out1, wh2) + bh2
	h_out2 = sigmoid(h_in2)


	output_in = np.dot( h_out2, wo) + bo
	output = sigmoid(output_in)

	#backpropagation
	e = error(output, y)
	d_out = out_grad(e, h_out2)
	wo = wo - lr * d_out


	dh2 = hidden_grad(e, h_out2)
	d_h2 =  np.dot(h_out1, dh2)
	wh2 = wh2 - lr * d_h2

	dh1 = hidden_grad(e, h_out2)
	d_h1 = np.dot(X.T, dh1)
	wh1 = wh1 - lr * d_h1


	bo = bo - lr*bo_grad(e)
	bh2 = bh2 - lr*bh_grad(e, h_out2)
	bh1 = bh1 - lr*bh_grad(e, h_out1)


print(output, accuracy(output, y))

t = np.array([[0, 1], [0,0], [0,0], [1,1]])
t= np.dot(t, wh1) + bh1
t = sigmoid(t)

t = np.dot(t, wh2) + bh2
t = sigmoid(t)

t = np.dot(t, wo) + bo
t = sigmoid(t)

print(t)


