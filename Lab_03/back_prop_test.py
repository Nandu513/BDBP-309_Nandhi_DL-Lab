import numpy as np
import random

def soft_max(z):
    deno=sum(np.exp(i) for i in z)
    s=[(np.exp(i)/deno) for i in z]
    return s

def relu(z):
    x=np.array([i if i>0 else 0 for i in z])
    return x

def bias(n):
    bw=[]
    print(f"1. Type 'Y' if you want to generate bias values randomly\n2. Type 'N' if you want to generate bias values manually" )
    prompt=input("Enter 'Y/N': ")
    if prompt=='Y':
        for i in range(1,len(n)):
            b = np.array([round(random.uniform(0.01, 3), 2) for j in range(n[i])])
            bw.append(b)
        return bw
    elif prompt=='N':
        for i in range(1,len(n)):
            b = np.array([float(input(f"Enter the layer{i}, b{j+1} value: ")) for j in range(n[i])])
            bw.append(b)
        return bw
    else:
        print("Enter the correct prompt to proceed further")
        bias(n)

def feed_forward(v,l,n,w,b):
    a=[]
    x=v
    for i in range(l):
        z=np.dot(w[i],x)+b[i]
        x=relu(z)
        a.append(x)
        print(f"Activation values of layer{i+1}\n",x)
        if i==l-1:
            print(f"1. Type R - if you want to use ReLu activation function at output layer\n2. Type S - to use Soft max activation function at the output layer")
            prompt=input("Enter R/S: ")
            if prompt=="S":
                x=soft_max(z)
            elif prompt=="R":
                x=relu(z)
            else:
                print("Enter correct prompt")
                feed_forward(v,l,n,w,b)
    return a


def main():
    try:
        # x=int(input("Enter the total no of input x values you want to give: "))
        # l=int(input("Enter the no of hidden layers(last layer is considered as output layer) you wan to build: "))
        # v=inputs_val(x)
        # n=layers(x,l)
        # w=weights(l,n)
        n = [2, 2, 1]
        b=bias(n)
        l = 2
        w = [
            [[0.1, 0.2],
             [0.3, 0.1]],
            [[1],
             [-1]]
        ]

        v = np.array([1,0])
        res = feed_forward(v, l, n, w, b)
        print(f"Final result: ",res)

    except Exception as e:
        print(f"An error occurred: {e}")


def derivatives(a, x, b, y):
    c = np.where(a == 0, 0, 1)
    w = (a - y) * c * x

    b = np.where(a == 0, 0, a - y)

    return w, b

# def back_propagation(wi, bi):





if __name__ == "__main__":
    main()


