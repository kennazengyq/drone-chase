"""
Class for plotting two quadrotors

update pose takes in arguments for the positions of both drones
"""

from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt

class Multiple_Quad():
    def __init__(self, a_x=0, a_y=0, a_z=0,b_x=0, b_y=0, b_z=0, roll=0, pitch=0, yaw=0, size=0.25, show_animation=True):
       
        # Quadrotors
        self.a_p1 = np.array([size / 2, 0, 0, 1]).T
        self.a_p2 = np.array([-size / 2, 0, 0, 1]).T
        self.a_p3 = np.array([0, size / 2, 0, 1]).T
        self.a_p4 = np.array([0, -size / 2, 0, 1]).T

        self.b_p1 = np.array([size / 2, 0, 0, 1]).T
        self.b_p2 = np.array([-size / 2, 0, 0, 1]).T
        self.b_p3 = np.array([0, size / 2, 0, 1]).T
        self.b_p4 = np.array([0, -size / 2, 0, 1]).T

        self.a_x_data = []
        self.a_y_data = []
        self.a_z_data = []

        self.b_x_data = []
        self.b_y_data = []
        self.b_z_data = []

        self.show_animation = show_animation

        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            self.ax = fig.add_subplot(111, projection='3d')

        self.update_pose(a_x, a_y, a_z, b_x, b_y, b_z, roll, pitch, yaw)

    def update_pose(self, a_x, a_y, a_z, b_x, b_y, b_z, roll, pitch, yaw):
        self.a_x = a_x
        self.a_y = a_y
        self.a_z = a_z

        self.b_x = b_x
        self.b_y = b_y
        self.b_z = b_z

        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.a_x_data.append(a_x)
        self.a_y_data.append(a_y)
        self.a_z_data.append(a_z)

        if self.show_animation:
            self.plot()

    def b_transformation_matrix(self):
        x = self.b_x
        y = self.b_y
        z = self.b_z
        roll = 0
        pitch = 0
        yaw = 0
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll), z]
             ])

    def a_transformation_matrix(self):
        x = self.a_x
        y = self.a_y
        z = self.a_z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll), z]
             ])

    def plot(self):  # pragma: no cover
        a_T = self.a_transformation_matrix()
        b_T = self.b_transformation_matrix()

        a_p1_t = np.matmul(a_T, self.a_p1)
        a_p2_t = np.matmul(a_T, self.a_p2)
        a_p3_t = np.matmul(a_T, self.a_p3)
        a_p4_t = np.matmul(a_T, self.a_p4)

        b_p1_t = np.matmul(b_T, self.b_p1)
        b_p2_t = np.matmul(b_T, self.b_p2)
        b_p3_t = np.matmul(b_T, self.b_p3)
        b_p4_t = np.matmul(b_T, self.b_p4)

        plt.cla()

        self.ax.plot([a_p1_t[0], a_p2_t[0], a_p3_t[0], a_p4_t[0]],
                     [a_p1_t[1], a_p2_t[1], a_p3_t[1], a_p4_t[1]],
                     [a_p1_t[2], a_p2_t[2], a_p3_t[2], a_p4_t[2]], 'k.')
        self.ax.plot([a_p1_t[0], a_p2_t[0]], [a_p1_t[1], a_p2_t[1]],
                     [a_p1_t[2], a_p2_t[2]], 'm-')
        self.ax.plot([a_p3_t[0], a_p4_t[0]], [a_p3_t[1], a_p4_t[1]],
                     [a_p3_t[2], a_p4_t[2]], 'm-')
        
        self.ax.plot([b_p1_t[0], b_p2_t[0], b_p3_t[0], b_p4_t[0]],
                     [b_p1_t[1], b_p2_t[1], b_p3_t[1], b_p4_t[1]],
                     [b_p1_t[2], b_p2_t[2], b_p3_t[2], b_p4_t[2]], 'k.')
        self.ax.plot([b_p1_t[0], b_p2_t[0]], [b_p1_t[1], b_p2_t[1]],
                     [b_p1_t[2], b_p2_t[2]], 'c-')
        self.ax.plot([b_p3_t[0], b_p4_t[0]], [b_p3_t[1], b_p4_t[1]],
                     [b_p3_t[2], b_p4_t[2]], 'c-')

        self.ax.plot(self.a_x_data, self.a_y_data, self.a_z_data, 'b:')



        plt.xlim(0,10)
        plt.ylim(0,10)
        self.ax.set_zlim(0,10)

        plt.pause(0.001)
    
    def get_pos_a(self):
        a_pos = np.array([self.a_x, self.a_y, self.a_z])
        a = a_pos.astype(int)
        return (a)
    
    def get_pos_b(self):
        b_pos = np.array([self.b_x, self.b_y, self.b_z])
        b = b_pos.astype(int)
        return (b)
    