#We start by defining the needed librairies
import networkx as nx               #We will use networkx to  represent graphs
import matplotlib.pyplot as plt     #We will use matplotlib.pyplot to draw graphs
import numpy as np                  #We will use numpy to manipulate matrices

#At first, we define the edges of the grapgh and the corresponding widths
Edges = [("x1","x2",{"weight" : 1 }),("x1","x4",{"weight" : 1 }),("x2","x3",{"weight" : 1 }) ,
                                                                 
        ("x4","x3",{"weight" : 1 }),("x2","x4",{"weight" : 1 }),("x1","x2",{"weight" : 1 }),
        ("x4","x5",{"weight" : 1 }) , ("x5","x6",{"weight" : 1 }),("x6","x7",{"weight" : 1 }),
        ("x7","x8",{"weight" : 1 }),("x8","x10",{"weight" : 1 }),("x8","x9",{"weight" : 1 })]

#We define the three graphs that will hold the signals
R = nx.Graph()
S = nx.Graph()
G = nx.Graph()
R.add_edges_from(Edges) 
S.add_edges_from(Edges) 
G.add_edges_from(Edges)

#We associate to each node in each graph the corresponding signal attribute
def create_signal(pos , graph) :
    
    for i in range(1,11) : 
        if pos!= i : 
            graph.nodes["x"+str(i)]["signal"] = 0 
            
        else : 
             graph.nodes["x"+str(i)]["signal"] = 1

#We then define the color map the distinguish between the nodes
def color_map (color , graph ) : 
    color_map = []
    for node in graph.nodes(data = True) : 
        if node[1]["signal"] == 1 :
            color_map.append(color)
        else : 
            color_map.append('blue')
    return color_map 


 
create_signal(1 , R)
color_mapR = color_map('red', R)
create_signal(7, S)
color_mapS = color_map('yellow', S)
create_signal(6 , G)
color_mapG = color_map('green', G)

#we create a list that contains the positions for different nodes
positions = {"x1" : [1,1] , "x2" : [1,0], "x3" : [1,-1],"x4" : [2,0],"x5" : [3,0],"x6" : [4,0],"x7" : [5,0],"x8" : [6,0]
             ,"x9" : [8,1],"x10" : [8,-1]}

#The visual result
plt.figure(1)
nx.draw(G,positions , with_labels= True, node_color= color_mapG ,  node_size = 500)
plt.figure(2)
nx.draw(R,positions, with_labels= True, node_color= color_mapR ,  node_size = 500)
plt.figure(3)
nx.draw(S, positions, with_labels= True, node_color= color_mapS ,  node_size = 500)
plt.show()

#In this section we will apply the diffusion measure and the euclidien measure to compare the different signals 

#We define the constants of the problem
alpha = 0.2
I = np.identity(10)

#We define a function to return the adjacency matrix
def adjacency_matrix( Graph : nx.graph , A = np.zeros((10,10)) ) :
    for e in  Graph.edges : 
        i = int(e[0][1])
        j = int(e[1][1:])
        A[i-1][j-1] = 1
        A[j-1][i-1] = 1  
    return A

#We define a function to return the degree matrix
def degree_matrix(A) : 
    D = np.zeros((10,10))
    for i in range(1,11) : 
        d = 0
        for j in range(1,11) : 
            d = A[i-1][j-1]+ d
        D[i-1][i-1] = d 
    return D

#We define a function to return the Laplacien matrix
def Laplacien_matrix(A,D) : 
    return A-D 

#We create the needed matices
A  = adjacency_matrix(R)
D  = degree_matrix(A)
L  = Laplacien_matrix(A,D) 

#We define the three signals used in the problem
r = np.array([1,0,0,0,0,0,0,0,0,0])
g = np.array([0,0,0,0,0,1,0,0,0,0])
s = np.array([0,0,0,0,0,0,1,0,0,0])

#We define the euclidien norm 
def euclidien_measure(signal , order) : 
    n = signal.shape[0] 
    l = 0
    for i in range(n) :
        l = signal[i] ** order + l 
    l = l**(1/order)
    return l 

#we define the diffusion norm
def diffusion_measure(signal : np.array ,Alpha, laplacien : np.array , order) :
    d = euclidien_measure(np.dot(np.linalg.inv((I+alpha*L)),s) , order)
    return d 
"""
print(np.linalg.norm(np.dot(np.linalg.inv(I+alpha*L), r-s)))
print(np.linalg.norm(np.dot(np.linalg.inv(I+alpha*L), s-g)))
print(np.linalg.norm(np.dot(np.linalg.inv(I+alpha*L), r-g)))
"""
print(D)

#Showing results 

print("calculating the diffusion distances : " + "\n")
print("The diffusion distance between r and s is : " + str(diffusion_measure(r-s , alpha, L , 2 ))+ "\n" )
print("The diffusion distance between r and g is : " + str(diffusion_measure(r-g , alpha, L , 2 ))+ "\n" )
print("The diffusion distance between g and s is : " + str(diffusion_measure(g-s , alpha, L , 2 ))+ "\n" )
print("calculating the euclidien distances : " + "\n")
print("The euclidien distance between r and s is : " + str(euclidien_measure(r-s , 2 ))+ "\n" )
print("The euclidien distance between r and g is : " + str(euclidien_measure(r-s , 2 ))+ "\n" )
print("The euclidien distance between g and s is : " + str(euclidien_measure(g-s , 2 ))+ "\n" )


