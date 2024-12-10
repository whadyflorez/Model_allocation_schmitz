import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the matrix
matrix_1 = np.array([[     0.      ,   478555.42200344   ,   0.        ],
 [791172.9739086  ,     0.      ,   158234.59285586],
 [     0.      ,   154382.95910553   ,   0.        ]])



matrix = np.array([[     0.  ,       567061.61893431    ,  0.        ],
 [708827.02608546   ,   0.      ,   141765.40705496],
 [     0.     ,         0.00000054   ,   0.00008984]])




# Get the shape of the matrix
num_rows, num_cols = matrix.shape

# Generate coordinates for the bars
x = np.arange(num_cols)
y = np.arange(num_rows)
X, Y = np.meshgrid(x, y)

# Flatten the matrix values for the heights of the bars
Z = matrix.flatten()

# Define category names
categories_product = ['Plastic', 'Textile', 'Paper']
#categories_product = ['Paper', 'Wood', 'Plastic','Textil']

company_names = [
    'Medellín','Cali','Bogota'
]

# company_names = [
#     "Fabricato", "Crystal", "Laffayette", "Proquinal", "Supertex", 
#     "Textiles Pacífico", "Indugevi", "Familia", "Empacor", "Inducartón", 
#     "Cartones América", "Smurfit Kappa", "Duratex", "Arauco", "Primadera", 
#     "Acemar", "Mundo Maderas", "Inducolma", "Alico", "Plásticos Truher", 
#     "Flexo Spring", "Multidimensionales", "Carvajal Empaques", "Amcor Holdings"
# ]


# Create the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the box plot
ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(Z), 1, 1, Z)

# Set the x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(categories_product, fontsize=8)

# Set the y-axis labels
ax.set_yticks(y)
ax.set_yticklabels(company_names, fontsize=8)

# Remove z-axis numbers and line
ax.set_zticks([])
ax.w_zaxis.line.set_lw(0)

# Add labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

#ax.set_box_aspect(aspect = (1,6,1))


# Show the plot
plt.show()
