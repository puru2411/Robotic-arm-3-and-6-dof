"""
	code by purushotam kumar agrawal  {git --> PURU2411 }

	Inverse kinematics of 3 dof spherical manipulator

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import numpy as np

a1 = .5
a2 = .5
a3 = 2
a4 = 2


def create_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.set_autoscale_on(False)
    # fig.canvas.draw()
    # plt.show()
    return fig, ax


def update_plot(X, Y, Z,  fig, ax):
    X = np.reshape(X, (1, 5))
    Y = np.reshape(Y, (1, 5))
    Z = np.reshape(Z, (1, 5))
    ax.cla()
    ax.plot_wireframe(X, Y, Z)
    plt.draw()
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.set_autoscale_on(True)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(.001)
    # ax.cla()
    # ax.plot_wireframe(Z,Y,X,color='r')
    # plt.pause(.5)
    # ax.plot_wireframe(Z,Y,X,color='b')
    # plt.pause(3)
    plt.show()


def forwardKinematics(q1, q2, d):

	H0_1 = np.matrix([[np.cos(q1), 0,  np.sin(q1), a2*np.cos(q1) ],
			[np.sin(q1), 0, -np.cos(q1), a2*np.sin(q1) ],
			[0,          1,           0,            a1 ],
			[0,          0,           0,             1 ]])

	H1_2 = np.matrix([[-np.sin(q2), 0, np.cos(q2), 0 ],
			[ np.cos(q2), 0, np.sin(q2), 0 ],
			[0,          1,           0, 0 ],
			[0,          0,           0, 1 ]])

	H2_3 = np.matrix([[1, 0, 0,       0 ],
			[0, 1, 0,       0 ],
			[0, 0, 1, a3+a4+d ],
			[0, 0, 0,       1 ]])

	H0_2 = np.dot(H0_1, H1_2)
	H0_3 = np.dot(H0_2, H2_3)

	X = []; Y = []; Z = []

	X.append(0)
	X.append(0)
	X.append(H0_1[0, 3])
	X.append(H0_2[0, 3])
	X.append(H0_3[0, 3])

	Y.append(0)
	Y.append(0)
	Y.append(H0_1[1, 3])
	Y.append(H0_2[1, 3])
	Y.append(H0_3[1, 3])

	Z.append(0)
	Z.append(a1)
	Z.append(H0_1[2, 3])
	Z.append(H0_2[2, 3])
	Z.append(H0_3[2, 3])

	return X, Y, Z


def get_inv_kin_values(x, y, z):

	q1 = np.arctan2(y, x)

	z1 = z-a1
	xy1 = np.sqrt(x**2 + y**2) - a2

	q2 = np.arctan2(z1, xy1)

	d = np.sqrt(z1**2 + xy1**2) - a3 - a4

	return q1, q2, d


def main():

	px = 2; py = 2; pz = 0

	q1, q2, d = get_inv_kin_values(px, py, pz)

	print("q1 : ", q1)
	print("q2 : ", q2)
	print("d : ", d)


	X, Y, Z = forwardKinematics(q1, q2, d)
	print(X)
	print(Y)
	print(Z)

	fig, ax = create_plot()
	update_plot(X, Y, Z,  fig, ax)


if __name__ == "__main__":
    main()
