# code written by Purushotam Kumar Agrawal { git ---> PURU2411 }

from sympy import symbols, cos, sin, pi, simplify, pprint, tan, expand_trig, sqrt, trigsimp, atan2
from sympy.matrices import Matrix

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import numpy as np

# declaring the link length of the arm
a1 = 1.0
a2 = 1.0
a3 = 5.0
a4 = 5.0
a5 = 0.2

dt = 0.001  # choosing dt = 1 ms

################################################################################################################
################################################################################################################
# forward kinematics is to verify the output and plot the graph

def forward_kin(q1, q2, q3, q4, q5, q6):
    X = []
    Y = []
    Z = []

    # DH parameter table of the given robotic arm see figure
    DH = np.array([[q1, np.pi / 2, a2, a1],
                 [q2, 0, a3, 0],
                 [q3 + np.pi / 2, np.pi / 2, 0, 0],
                 [q4, -np.pi / 2, 0, a4],
                 [q5, np.pi / 2, 0, 0],
                 [q6, 0, 0, a5]])

    # homogeneous matrices
    H0_0 = Matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    H0_1 = Matrix([[np.cos(DH[0, 0]), -np.sin(DH[0, 0]) * np.cos(DH[0, 1]), np.sin(DH[0, 0]) * np.sin(DH[0, 1]), DH[0, 2] * np.cos(DH[0, 0])],
                   [np.sin(DH[0, 0]), np.cos(DH[0, 0]) * np.cos(DH[0, 1]), -np.cos(DH[0, 0]) * np.sin(DH[0, 1]), DH[0, 2] * np.sin(DH[0, 0])],
                   [0, np.sin(DH[0, 1]), np.cos(DH[0, 1]), DH[0, 3]],
                   [0, 0, 0, 1]])

    H1_2 = Matrix([[np.cos(DH[1, 0]), -np.sin(DH[1, 0]) * np.cos(DH[1, 1]), np.sin(DH[1, 0]) * np.sin(DH[1, 1]), DH[1, 2] * np.cos(DH[1, 0])],
                   [np.sin(DH[1, 0]), np.cos(DH[1, 0]) * np.cos(DH[1, 1]), -np.cos(DH[1, 0]) * np.sin(DH[1, 1]), DH[1, 2] * np.sin(DH[1, 0])],
                   [0, np.sin(DH[1, 1]), np.cos(DH[1, 1]), DH[1, 3]],
                   [0, 0, 0, 1]])

    H0_2 = np.dot(H0_1, H1_2)

    H2_3 = Matrix([[np.cos(DH[2, 0]), -np.sin(DH[2, 0]) * np.cos(DH[2, 1]), np.sin(DH[2, 0]) * np.sin(DH[2, 1]), DH[2, 2] * np.cos(DH[2, 0])],
                   [np.sin(DH[2, 0]), np.cos(DH[2, 0]) * np.cos(DH[2, 1]), -np.cos(DH[2, 0]) * np.sin(DH[2, 1]), DH[2, 2] * np.sin(DH[2, 0])],
                   [0, np.sin(DH[2, 1]), np.cos(DH[2, 1]), DH[2, 3]],
                   [0, 0, 0, 1]])

    H0_3 = np.dot(H0_2, H2_3)

    H3_4 = Matrix([[np.cos(DH[3, 0]), -np.sin(DH[3, 0]) * np.cos(DH[3, 1]), np.sin(DH[3, 0]) * np.sin(DH[3, 1]), DH[3, 2] * np.cos(DH[3, 0])],
                   [np.sin(DH[3, 0]), np.cos(DH[3, 0]) * np.cos(DH[3, 1]), -np.cos(DH[3, 0]) * np.sin(DH[3, 1]), DH[3, 2] * np.sin(DH[3, 0])],
                   [0, np.sin(DH[3, 1]), np.cos(DH[3, 1]), DH[3, 3]],
                   [0, 0, 0, 1]])

    H0_4 = np.dot(H0_3, H3_4)

    H4_5 = Matrix([[np.cos(DH[4, 0]), -np.sin(DH[4, 0]) * np.cos(DH[4, 1]), np.sin(DH[4, 0]) * np.sin(DH[4, 1]), DH[4, 2] * np.cos(DH[4, 0])],
                   [np.sin(DH[4, 0]), np.cos(DH[4, 0]) * np.cos(DH[4, 1]), -np.cos(DH[4, 0]) * np.sin(DH[4, 1]), DH[4, 2] * np.sin(DH[4, 0])],
                   [0, np.sin(DH[4, 1]), np.cos(DH[4, 1]), DH[4, 3]],
                   [0, 0, 0, 1]])

    H0_5 = np.dot(H0_4, H4_5)

    H5_6 = Matrix([[np.cos(DH[5, 0]), -np.sin(DH[5, 0]) * np.cos(DH[5, 1]), np.sin(DH[5, 0]) * np.sin(DH[5, 1]), DH[5, 2] * np.cos(DH[5, 0])],
                   [np.sin(DH[5, 0]), np.cos(DH[5, 0]) * np.cos(DH[5, 1]), -np.cos(DH[5, 0]) * np.sin(DH[5, 1]), DH[5, 2] * np.sin(DH[5, 0])],
                   [0, np.sin(DH[5, 1]), np.cos(DH[5, 1]), DH[5, 3]],
                   [0, 0, 0, 1]])

    H0_6 = np.dot(H0_5, H5_6)

    # print("R0_6 comes out to be: ")
    # print(np.matrix(H0_6[:3, :3]))

    X.append(0)
    X.append(0)
    X.append(H0_1[0, 3])
    X.append(H0_2[0, 3])
    X.append(H0_3[0, 3])
    X.append(H0_4[0, 3])
    X.append(H0_5[0, 3])
    X.append(H0_6[0, 3])

    Y.append(0)
    Y.append(0)
    Y.append(H0_1[1, 3])
    Y.append(H0_2[1, 3])
    Y.append(H0_3[1, 3])
    Y.append(H0_4[1, 3])
    Y.append(H0_5[1, 3])
    Y.append(H0_6[1, 3])

    Z.append(0)
    Z.append(a1)
    Z.append(H0_1[2, 3])
    Z.append(H0_2[2, 3])
    Z.append(H0_3[2, 3])
    Z.append(H0_4[2, 3])
    Z.append(H0_5[2, 3])
    Z.append(H0_6[2, 3])

    # center of all the frames in ground frame
    X = np.reshape(X, (1, 8))
    Y = np.reshape(Y, (1, 8))
    Z = np.reshape(Z, (1, 8))

    return X, Y, Z



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


def update_plot(X, Y, Z, X1, Y1, Z1,  fig, ax):
    X = np.reshape(X, (1, 8))
    Y = np.reshape(Y, (1, 8))
    Z = np.reshape(Z, (1, 8))
    ax.cla()
    ax.plot_wireframe(X, Y, Z)
    ax.plot_wireframe(X1, Y1, Z1, color = 'r')
    # print('in update ', X1, Y1, Z1)
    plt.draw()
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.set_autoscale_on(False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(.001)
    # ax.cla()
    # ax.plot_wireframe(Z,Y,X,color='r')
    # plt.pause(.5)
    # ax.plot_wireframe(Z,Y,X,color='b')
    # plt.pause(3)
    # plt.show()

################################################################################################################
################################################################################################################


# this function is used only to calculate some of the matrices used in it's original trigonometric form like R0_3 and R3_6
def printMatrices():
    a, b, c = symbols('alpha beta gama', real=True)
    # rotation matrix after rotating around y-axis (pitch)
    A = Matrix([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])

    # rotation matrix after rotating around z-axis (roll)
    B = Matrix([[cos(b), -sin(b), 0], [sin(b), cos(b), 0], [0, 0, 1]])

    # rotation matrix after rotating around x-axis (yaw)
    C = Matrix([[1, 0, 0], [0, cos(c), -sin(c)], [0, sin(c), cos(c)]])

    # after total rotation of pitch, roll and yaw
    D = A * B * C
    print(D)

    q1, q2, q3, q4, q5, q6 = symbols('q1:7')
    a1, a2, a3, a4, a5, a6, a7 = symbols('a1:8')

    # DH parameter of the given arm
    DH = Matrix([[q1, pi/2, a2, a1],
                [q2, 0,    a3, 0 ],
                [q3+pi/2, pi/2, 0, 0],
                [q4, -pi/2, 0, a4],
                [q5, pi/2, 0, 0],
                [q6, 0, 0, a5]])

    # homogeneous matrices
    H0_1 = Matrix([[cos(DH[0,0]), -sin(DH[0,0])*cos(DH[0,1]), sin(DH[0,0])*sin(DH[0,1]), DH[0,2]*cos(DH[0,0])],
                  [sin(DH[0,0]), cos(DH[0,0])*cos(DH[0,1]), -cos(DH[0,0])*sin(DH[0,1]), DH[0,2]*sin(DH[0,0])],
                  [0,            sin(DH[0,1]),               cos(DH[0,1]),              DH[0,3]             ],
                  [0,            0,                          0,                         1                   ]])

    H1_2 = Matrix([[cos(DH[1,0]), -sin(DH[1,0])*cos(DH[1,1]), sin(DH[1,0])*sin(DH[1,1]), DH[1,2]*cos(DH[1,0])],
                  [sin(DH[1,0]), cos(DH[1,0])*cos(DH[1,1]), -cos(DH[1,0])*sin(DH[1,1]), DH[1,2]*sin(DH[1,0])],
                  [0,            sin(DH[1,1]),               cos(DH[1,1]),              DH[1,3]             ],
                  [0,            0,                          0,                         1                   ]])

    H2_3 = Matrix([[cos(DH[2,0]), -sin(DH[2,0])*cos(DH[2,1]), sin(DH[2,0])*sin(DH[2,1]), DH[2,2]*cos(DH[2,0])],
                  [sin(DH[2,0]), cos(DH[2,0])*cos(DH[2,1]), -cos(DH[2,0])*sin(DH[2,1]), DH[2,2]*sin(DH[2,0])],
                  [0,            sin(DH[2,1]),               cos(DH[2,1]),              DH[2,3]             ],
                  [0,            0,                          0,                         1                   ]])

    H3_4 = Matrix([[cos(DH[3,0]), -sin(DH[3,0])*cos(DH[3,1]), sin(DH[3,0])*sin(DH[3,1]), DH[3,2]*cos(DH[3,0])],
                  [sin(DH[3,0]), cos(DH[3,0])*cos(DH[3,1]), -cos(DH[3,0])*sin(DH[3,1]), DH[3,2]*sin(DH[3,0])],
                  [0,            sin(DH[3,1]),               cos(DH[3,1]),              DH[3,3]             ],
                  [0,            0,                          0,                         1                   ]])

    H4_5 = Matrix([[cos(DH[4,0]), -sin(DH[4,0])*cos(DH[4,1]), sin(DH[4,0])*sin(DH[4,1]), DH[4,2]*cos(DH[4,0])],
                  [sin(DH[4,0]), cos(DH[4,0])*cos(DH[4,1]), -cos(DH[4,0])*sin(DH[4,1]), DH[4,2]*sin(DH[4,0])],
                  [0,            sin(DH[4,1]),               cos(DH[4,1]),              DH[4,3]             ],
                  [0,            0,                          0,                         1                   ]])

    H5_6 = Matrix([[cos(DH[5,0]), -sin(DH[5,0])*cos(DH[5,1]), sin(DH[5,0])*sin(DH[5,1]), DH[5,2]*cos(DH[5,0])],
                  [sin(DH[5,0]), cos(DH[5,0])*cos(DH[5,1]), -cos(DH[5,0])*sin(DH[5,1]), DH[5,2]*sin(DH[5,0])],
                  [0,            sin(DH[5,1]),               cos(DH[5,1]),              DH[5,3]             ],
                  [0,            0,                          0,                         1                   ]])

    H0_6 = H0_1*H1_2*H2_3*H3_4*H4_5*H5_6
    print(H0_6)

    print(H0_1)
    print(H1_2)
    print(H2_3)
    print(H3_4)
    print(H4_5)
    print(H5_6)

    # rotation matrices
    R0_1 = H0_1[:3, :3]
    R0_2 = R0_1*H1_2[:3, :3]
    R0_3 = R0_2*H2_3[:3, :3]
    R0_4 = R0_3*H3_4[:3, :3]
    R0_5 = R0_4*H4_5[:3, :3]
    R0_6 = R0_5*H5_6[:3, :3]
    print(R0_1)
    print(R0_2)
    print(R0_3)
    print(R0_4)
    print(R0_5)
    print(R0_6)

    R36 = H3_4[:3, :3]*H4_5[:3, :3]*H5_6[:3, :3]
    print(R36)


def get_cosine_law_angle(a, b, c):
    # given all sides of a triangle a, b, c
    # calculate angle gamma between sides a and b using cosine law

    gamma = np.arccos((a*a + b*b - c*c) / (2*a*b))

    return gamma


def griperCenter(px, py, pz, R06):
    # calculating griper center, see in arm diagram for detail
    Xc = px - a5*R06[0,2]
    Yc = py - a5*R06[1,2]
    Zc = pz - a5*R06[2,2]
    return Xc, Yc, Zc


def calcFirst3Angles(Xc, Yc, Zc):
    # doing inverse kinematics on first 3 dof the reach the center of griper
    # see the calculation page of inverse kinematics for more details

    q1 = np.arctan2(Yc, Xc)

    r1 = np.sqrt(Xc**2 + Yc**2)
    r2 = np.sqrt((r1-a2)**2 + (Zc-a1)**2)

    phi1 = np.arctan((Zc-a1)/(r1-a2))
    phi2 = get_cosine_law_angle(a3, r2, a4)
    q2 = phi1 + phi2

    phi3 = get_cosine_law_angle(a3, a4, r2)
    q3 = phi3 - np.pi

    return q1, q2, q3


def calcLast3Angles(R36):
    # evaluating last 3 angles by comparing the matrices
    # R36 = Matrix([[-sin(q4)*sin(q6) + cos(q4)*cos(q5)*cos(q6), -sin(q4)*cos(q6) - sin(q6)*cos(q4)*cos(q5), sin(q5)*cos(q4)],
    #               [sin(q4)*cos(q5)*cos(q6) + sin(q6)*cos(q4), -sin(q4)*sin(q6)*cos(q5) + cos(q4)*cos(q6), sin(q4)*sin(q5)],
    #               [-sin(q5)*cos(q6)                         , sin(q5)*sin(q6)                           , cos(q5)]])

    q4 = np.arctan2(R36[1,2],R36[0, 2])

    q5 = np.arccos(R36[2,2])

    q6 = np.arctan2(R36[2,1],-R36[2,0])
    return q4, q5, q6


def get_angles(px, py, pz, beta, alpha, gama):

    # the frame of griper is pre-rotated from bellow rotation matrix
    R6a = [[0, 0, 1.0], [0, -1.0, 0], [1.0, 0, 0]]

    # after rotation of pitch, roll, yaw
    R6b = [[np.cos(alpha)*np.cos(beta), np.sin(alpha)*np.sin(gama) - np.sin(beta)*np.cos(alpha)*np.cos(gama), np.sin(alpha)*np.cos(gama) + np.sin(beta)*np.sin(gama)*np.cos(alpha)],
           [np.sin(beta)           , np.cos(beta)*np.cos(gama)                                  , -np.sin(gama)*np.cos(beta)],
           [-np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.cos(gama) + np.sin(gama)*np.cos(alpha), -np.sin(alpha)*np.sin(beta)*np.sin(gama) + np.cos(alpha)*np.cos(gama)]]
    # total rotation of griper frame WRT ground frame
    R06 = np.dot(R6a,R6b)
    # print(np.matrix(R06))

    # calculating center of griper
    Xc, Yc, Zc = griperCenter(px, py, pz, R06)

    # calculating first 3 angles
    q1, q2, q3 = calcFirst3Angles(Xc, Yc, Zc)

    # rotation matrix of 3 wrt 0 frame  see the calculation sheet for more understanding
    R03 = [[-np.sin(q2) * np.cos(q1) * np.cos(q3) - np.sin(q3) * np.cos(q1) * np.cos(q2), np.sin(q1), -np.sin(q2) * np.sin(q3) * np.cos(q1) + np.cos(q1) * np.cos(q2) * np.cos(q3)],
           [-np.sin(q1) * np.sin(q2) * np.cos(q3) - np.sin(q1) * np.sin(q3) * np.cos(q2), -np.cos(q1), -np.sin(q1) * np.sin(q2) * np.sin(q3) + np.sin(q1) * np.cos(q2) * np.cos(q3)],
           [-np.sin(q2) * np.sin(q3) + np.cos(q2) * np.cos(q3), 0, np.sin(q2) * np.cos(q3) + np.sin(q3) * np.cos(q2)]]

    IR03 = np.transpose(R03)

    R36 = np.dot(IR03, R06)

    q4, q5, q6 = calcLast3Angles(R36)

    return q1, q2, q3, q4, q5, q6


def main():

    # printMatrices()

    # position of end effector
    px, py, pz = 5.0, 1.0, 1.0
    # value of orientation of the end effector
    roll, pitch, yaw = 0, 0, 0

    q1, q2, q3, q4, q5, q6 = get_angles(px, py, pz, roll, pitch, yaw)

    print("q1 : ", q1)
    print("q2 : ", q2)
    print("q3 : ", q3)
    print("q4 : ", q4)
    print("q5 : ", q5)
    print("q6 : ", q6)

    # to check the output
    X, Y, Z = forward_kin(q1, q2, q3, q4, q5, q6)
    print("X : ", X[0, 7])
    print("Y : ", Y[0, 7])
    print("Z : ", Z[0, 7])

    fig, ax = create_plot()
    update_plot(X, Y, Z, X, Y, Z, fig, ax)
    plt.show()


if __name__ == "__main__":
    main()

