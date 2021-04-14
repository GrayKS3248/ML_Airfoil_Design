# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:43:28 2021

@author: Grayson Schaer
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

class Panels:
    
    def __init__(self, n_panels):
        
        self.n_panels = self.get_n_panels(n_panels)
        
        self.x_coords, self.y_coords, self.camber_line = self.get_coords(self.n_panels)
        self.control_x_coords, self.control_y_coords = self.get_points(self.x_coords, self.y_coords)
        self.normal = self.get_normal(self.x_coords, self.y_coords)
        self.lengths = self.get_length(self.x_coords, self.y_coords)
        self.theta = self.get_angles(self.x_coords, self.y_coords, self.lengths)
        
    # Allows user to set y coords of panels
    # @param y coords to be set
    def set_y_coords(self, y_coords):
        self.y_coords = y_coords
        self.camber_line = self.get_camber(self.y_coords)
        self.control_x_coords, self.control_y_coords = self.get_points(self.x_coords, self.y_coords)
        self.normal = self.get_normal(self.x_coords, self.y_coords)
        self.lengths = self.get_length(self.x_coords, self.y_coords)
        self.theta = self.get_angles(self.x_coords, self.y_coords, self.lengths)
        
    # Calculates the camberline of given panels coordinates
    # @param Y cooridinates of panels
    def get_camber(self, y_coords):
        
        bot_surface = y_coords[0:len(y_coords)//2+1]
        top_surface = np.flip(y_coords[len(y_coords)//2:])
        camber_line = (top_surface + bot_surface) / 2.0
        
        return camber_line
        
    # Ensures the passed number of panels is valid
    # @param Number of panels to create
    def get_n_panels(self, n_panels):
        
        if int(round(n_panels)) % 2 == 0:
            return int(round(n_panels))
        
        else:
            raise Exception("Invalid number of panels (must be even).")
        
    # Gets the x/c and y/c normalized coordinates of the panels
    # @param Number of panels
    def get_coords(self, n_panels):
        
        x_coords = self.get_x_coords(n_panels)
        y_coords, camber_line = self.get_y_coords(x_coords)
        return x_coords, y_coords, camber_line
    
    # Gets the x/c normalized coordinates of the panels
    # @param Number of panels
    def get_x_coords(self, n_panels):
        
        n = (n_panels//2)
        j = np.arange(n+1)
        top_coords = 0.5 - 0.5*np.cos(j*np.pi/n)
        bot_coords = 0.5 + 0.5*np.cos(j*np.pi/n)
        x_coords = np.concatenate((bot_coords, top_coords[1:]))
        
        return x_coords

    # Gets the y/c normalized coordinates of the panels and camber updated x/c normalized coords of the panels
    # @param X cooridinates of panels
    def get_y_coords(self, x_coords):
        
        x_on_c = x_coords[0:len(x_coords)//2+1]
        
        max_thickness = 0.15 * np.random.rand() + 0.10
        max_thickness_loc = 0.10 * np.random.rand() + 0.05
        LE_thickness_pitch = 1.0 * np.random.rand() + 1.0
        TE_thickness_pitch = 1.0 * np.random.rand() + 1.0
        
        LE_term = LE_thickness_pitch*np.exp(np.log((LE_thickness_pitch+1.0)/LE_thickness_pitch)*((2.0*x_on_c)/max_thickness_loc - (x_on_c/max_thickness_loc)**2.0))-LE_thickness_pitch
        TE_term = TE_thickness_pitch*((TE_thickness_pitch+1.0)/TE_thickness_pitch)**(((2.0*max_thickness_loc-x_on_c-1.0)*(x_on_c-1.0))/(max_thickness_loc-1.0)**2.0)-TE_thickness_pitch
        
        LE_half_thickness = 0.5*max_thickness*LE_term*(x_on_c<=max_thickness_loc)
        TE_half_thickness = 0.5*max_thickness*TE_term*(x_on_c>max_thickness_loc)
        half_thickness = LE_half_thickness + TE_half_thickness
        half_thickness[half_thickness<=1.0e-8] = 0.0
        
        camber_type = np.random.randint(0,4)
        
        if camber_type==0:
            max_camber = 0.069 * np.random.rand() + 0.001
            max_camber_loc = 0.40 * np.random.rand() + 0.10
            LE_camber_pitch = 1.0 * np.random.rand() + 1.0
            TE_camber_pitch = 1.99 * np.random.rand() + 0.01
            
            LE_term = LE_camber_pitch*np.exp(np.log((LE_camber_pitch+1.0)/LE_camber_pitch)*((2.0*x_on_c)/max_camber_loc - (x_on_c/max_camber_loc)**2.0))-LE_camber_pitch
            TE_term = TE_camber_pitch*((TE_camber_pitch+1.0)/TE_camber_pitch)**(((2.0*max_camber_loc-x_on_c-1.0)*(x_on_c-1.0))/(max_camber_loc-1.0)**2.0)-TE_camber_pitch
            
            LE_camber = max_camber*LE_term*(x_on_c<=max_camber_loc)
            TE_camber = max_camber*TE_term*(x_on_c>max_camber_loc)
            camber_line = LE_camber + TE_camber
            camber_line[camber_line<=1.0e-8] = 0.0
        
        elif camber_type==1:
            camber_inflection = 0.40 * np.random.rand() + 0.50
            
            LE_max_camber = 0.069 * np.random.rand() + 0.001
            LE_max_camber_loc = 0.30 * camber_inflection  * np.random.rand() + 0.30 * camber_inflection
            LE_camber_pitch = 1.0 * np.random.rand() + 1.0
            LE_inflection_pitch = 1.99 * np.random.rand() + 0.01
            
            LE_sigma_1 = (np.sqrt(2.0)*LE_max_camber_loc) / (2.0*np.sqrt(np.log((LE_camber_pitch+1.0)/LE_camber_pitch)))
            LE_term_1 = (LE_camber_pitch + 1.0)*np.exp(-0.5*((x_on_c-LE_max_camber_loc)/LE_sigma_1)**2.0)-LE_camber_pitch
            LE_sigma_2 = (np.sqrt(2.0)*(camber_inflection-LE_max_camber_loc)) / (2.0*np.sqrt(np.log((LE_inflection_pitch+1.0)/LE_inflection_pitch)))
            LE_term_2 = (LE_inflection_pitch + 1.0)*np.exp(-0.5*((x_on_c-LE_max_camber_loc)/LE_sigma_2)**2.0)-LE_inflection_pitch
            
            LE_camber_1 = LE_max_camber*LE_term_1*(x_on_c<=LE_max_camber_loc)
            LE_camber_2 = LE_max_camber*LE_term_2*((x_on_c>LE_max_camber_loc) * (x_on_c<=camber_inflection))
            LE_camber = LE_camber_1 + LE_camber_2
            
            TE_max_camber = 0.049 * np.random.rand() + 0.001
            TE_max_camber_loc = (0.90 - camber_inflection - 0.1)  * np.random.rand() + camber_inflection + 0.1
            TE_inflection_pitch = 1.99 * np.random.rand() + 0.01
            TE_camber_pitch = 1.0 * np.random.rand() + 1.0
            
            TE_sigma_1 = (np.sqrt(2.0)*(TE_max_camber_loc-camber_inflection)) / (2.0*np.sqrt(np.log((TE_inflection_pitch+1.0)/TE_inflection_pitch)))
            TE_term_1 = (TE_inflection_pitch + 1.0)*np.exp(-0.5*((x_on_c-TE_max_camber_loc)/TE_sigma_1)**2.0)-TE_inflection_pitch
            TE_sigma_2 = (np.sqrt(2.0)*(1.0-TE_max_camber_loc)) / (2.0*np.sqrt(np.log((TE_camber_pitch+1.0)/TE_camber_pitch)))
            TE_term_2 = (TE_camber_pitch + 1.0)*np.exp(-0.5*((x_on_c-TE_max_camber_loc)/TE_sigma_2)**2.0)-TE_camber_pitch
            
            TE_camber_1 = -TE_max_camber*TE_term_1*((x_on_c>camber_inflection) * (x_on_c<=TE_max_camber_loc))
            TE_camber_2 = -TE_max_camber*TE_term_2*((x_on_c>TE_max_camber_loc))
            TE_camber = TE_camber_1 + TE_camber_2
            
            camber_line = LE_camber + TE_camber
            camber_line[abs(camber_line)<=1.0e-8] = 0.0
        
        elif camber_type==2:
            camber_inflection = 0.30 * np.random.rand() + 0.20
            
            LE_max_camber = 0.069 * np.random.rand() + 0.001
            LE_max_camber_loc = 0.30 * camber_inflection  * np.random.rand() + 0.30 * camber_inflection
            LE_camber_pitch = 1.0 * np.random.rand() + 1.0
            LE_inflection_pitch = 1.99 * np.random.rand() + 0.01
            
            LE_sigma_1 = (np.sqrt(2.0)*LE_max_camber_loc) / (2.0*np.sqrt(np.log((LE_camber_pitch+1.0)/LE_camber_pitch)))
            LE_term_1 = (LE_camber_pitch + 1.0)*np.exp(-0.5*((x_on_c-LE_max_camber_loc)/LE_sigma_1)**2.0)-LE_camber_pitch
            LE_sigma_2 = (np.sqrt(2.0)*(camber_inflection-LE_max_camber_loc)) / (2.0*np.sqrt(np.log((LE_inflection_pitch+1.0)/LE_inflection_pitch)))
            LE_term_2 = (LE_inflection_pitch + 1.0)*np.exp(-0.5*((x_on_c-LE_max_camber_loc)/LE_sigma_2)**2.0)-LE_inflection_pitch
            
            LE_camber_1 = -LE_max_camber*LE_term_1*(x_on_c<=LE_max_camber_loc)
            LE_camber_2 = -LE_max_camber*LE_term_2*((x_on_c>LE_max_camber_loc) * (x_on_c<=camber_inflection))
            LE_camber = LE_camber_1 + LE_camber_2
            
            TE_max_camber = 0.049 * np.random.rand() + 0.001
            TE_max_camber_loc = (0.90 - camber_inflection - 0.1)  * np.random.rand() + camber_inflection + 0.1
            TE_inflection_pitch = 1.99 * np.random.rand() + 0.01
            TE_camber_pitch = 1.0 * np.random.rand() + 1.0
            
            TE_sigma_1 = (np.sqrt(2.0)*(TE_max_camber_loc-camber_inflection)) / (2.0*np.sqrt(np.log((TE_inflection_pitch+1.0)/TE_inflection_pitch)))
            TE_term_1 = (TE_inflection_pitch + 1.0)*np.exp(-0.5*((x_on_c-TE_max_camber_loc)/TE_sigma_1)**2.0)-TE_inflection_pitch
            TE_sigma_2 = (np.sqrt(2.0)*(1.0-TE_max_camber_loc)) / (2.0*np.sqrt(np.log((TE_camber_pitch+1.0)/TE_camber_pitch)))
            TE_term_2 = (TE_camber_pitch + 1.0)*np.exp(-0.5*((x_on_c-TE_max_camber_loc)/TE_sigma_2)**2.0)-TE_camber_pitch
            
            TE_camber_1 = TE_max_camber*TE_term_1*((x_on_c>camber_inflection) * (x_on_c<=TE_max_camber_loc))
            TE_camber_2 = TE_max_camber*TE_term_2*((x_on_c>TE_max_camber_loc))
            TE_camber = TE_camber_1 + TE_camber_2
            
            camber_line = LE_camber + TE_camber
            camber_line[abs(camber_line)<=1.0e-8] = 0.0
        
        elif camber_type==3:
            max_camber = 0.069 * np.random.rand() + 0.001
            max_camber_loc = 0.40 * np.random.rand() + 0.10
            LE_camber_pitch = 1.0 * np.random.rand() + 1.0
            TE_camber_pitch = 1.99 * np.random.rand() + 0.01
            
            LE_term = LE_camber_pitch*np.exp(np.log((LE_camber_pitch+1.0)/LE_camber_pitch)*((2.0*x_on_c)/max_camber_loc - (x_on_c/max_camber_loc)**2.0))-LE_camber_pitch
            TE_term = TE_camber_pitch*((TE_camber_pitch+1.0)/TE_camber_pitch)**(((2.0*max_camber_loc-x_on_c-1.0)*(x_on_c-1.0))/(max_camber_loc-1.0)**2.0)-TE_camber_pitch
            
            LE_camber = -max_camber*LE_term*(x_on_c<=max_camber_loc)
            TE_camber = -max_camber*TE_term*(x_on_c>max_camber_loc)
            camber_line = LE_camber + TE_camber
            camber_line[abs(camber_line)<=1.0e-8] = 0.0
        
        y_upper = camber_line + half_thickness
        y_lower = camber_line - half_thickness
            
        y_coords = np.concatenate((y_lower, np.flip(y_upper)[1:]))
        y_coords[0] = 0.0
        y_coords[-1] = 0.0
        
        return y_coords, camber_line
    
    # Gets the locations of the vortices and control points
    # @param X coords of panels
    # @param Y coords of panels
    def get_points(self, x_coords, y_coords):
        
        control_x_coords = x_coords[1:]-0.5*np.diff(x_coords)
        control_y_coords = y_coords[1:]-0.5*np.diff(y_coords)
        
        return control_x_coords, control_y_coords
    
    # Solve the normal vectors for each panel
    # @param X coords of panels
    # @param Y coords of panels
    def get_normal(self, x_coords, y_coords):
        
        x_dirn = np.diff(x_coords).reshape(len(x_coords)-1,1)
        y_dirn = np.diff(y_coords).reshape(len(y_coords)-1,1)
        
        tangent = np.transpose(np.concatenate((x_dirn, y_dirn), axis=1))
        rotation = np.array([[0.0, -1.0],[1.0, 0.0]])
       
        normal = np.matmul(rotation, tangent)
        normal = normal / np.sqrt(normal[0,:]**2.0 + normal[1,:]**2.0)
        return normal
    
    # Solve the length of each panel
    # @param X coords of panels
    # @param Y coords of panels
    def get_length(self, x_coords, y_coords):
        
        lengths = (np.diff(y_coords)**2.0+np.diff(x_coords)**2.0)**0.50
        return lengths
    
    # Solves the orientation angle between each panel and the x-axis
    # @param X coords of panels
    # @param Y coords of panels
    # @param Length of each panel
    def get_angles(self, x_coords, y_coords, lengths):
        
        theta = np.arctan2(np.diff(y_coords), np.diff(x_coords))
        return theta
    
    # Renders and save the panels
    # @param save path
    # @param name of airfoil
    def draw(self, path, airfoil_name=''):

        if not os.path.isdir(path):
            os.mkdir(path)
        
        num = 0
        done = False
        while not done:
            done = not (os.path.exists(path + "/airfoil_" + str(num) + ".png"))
            num = num + 1
        num = num - 1
        
        if 'rebuilt' in airfoil_name.lower():
            path = path + "/airfoil_" + str(num-1) + "_rebuilt.png"
        else:
            path = path + "/airfoil_" + str(num) + ".png"
        
        plt.close()
        
        normal_x_coords_start = self.x_coords[1:]-0.5*np.diff(self.x_coords)
        normal_y_coords_start = self.y_coords[1:]-0.5*np.diff(self.y_coords)
        normal_x_coords_end = normal_x_coords_start + 0.04 * self.normal[0,:]
        normal_y_coords_end = normal_y_coords_start + 0.04 * self.normal[1,:]
        
        plt.plot(self.x_coords[0:len(self.x_coords)//2+1], self.camber_line, lw=2.0, ls="--", c='r')
        plt.axhline(0.0, lw=2.0, c='r')
        plt.plot(self.x_coords, self.y_coords, color='b', lw=2.0)
        plt.plot(self.control_x_coords, self.control_y_coords, 'd', color='g', markersize=7)
        plt.plot(self.x_coords, self.y_coords, 'o', color='k', markersize=7)
        
        for i in range(len(normal_x_coords_start)):
            
            x_points = np.array([normal_x_coords_start[i],normal_x_coords_end[i]])
            y_points = np.array([normal_y_coords_start[i],normal_y_coords_end[i]])
            plt.plot(x_points, y_points, color='k', lw=2.0)
            
            y_diff = np.diff(y_points)
            x_diff = np.diff(x_points)
            tangent = np.arctan(y_diff / x_diff)[0]*180.0/np.pi
            
            if x_diff >= 0.0 and y_diff >= 0.0:
                angle = -(90.0 - tangent)
            elif x_diff < 0.0 and y_diff > 0.0:
                angle = (90.0 - abs(tangent))
            elif x_diff < 0.0 and y_diff < 0.0:
                angle = -(90.0 - tangent) + 180.0
            elif x_diff > 0.0 and y_diff < 0.0:
                angle = (90.0 - abs(tangent)) + 180.0
                
            t = mpl.markers.MarkerStyle(marker='^')
            t._transform = t.get_transform().rotate_deg(angle)
            plt.plot(normal_x_coords_end[i], normal_y_coords_end[i], marker=t, color='k', markersize=8)
        
        plt.xlabel("X/C [-]", fontsize="large")
        plt.ylabel("Y/C [-]", fontsize="large")
        if airfoil_name == '':
            plt.title('Airfoil', fontsize="xx-large")
        else:
            plt.title(airfoil_name, fontsize="xx-large")
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.25, 0.20])
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize='large')
        plt.yticks([-0.25, -0.15, -0.05, 0.05, 0.15, 0.25],fontsize='large')
        
        plt.gcf().set_size_inches(8,2.8)
        plt.savefig(path, dpi=200)
        
class Solver:
    
    def __init__(self):
        self.panels=0.0
        self.alpha=0.0
        self.v_panels=0.0
        self.cp=0.0
        self.Cl=0.0
        self.Cdp=0.0
        self.Cmc4=0.0
    
    # Solves the total local velocity at each control point based on linear varying vortex panel method
    # @param Angle of attack
    # @param Panels object that defines airfoil geometry
    def get_velocity_vp(self, alpha, panels):

        Cn1 = np.zeros((len(panels.control_x_coords),len(panels.control_x_coords)))
        Cn2 = np.zeros((len(panels.control_x_coords),len(panels.control_x_coords)))
        Ct1 = np.zeros((len(panels.control_x_coords),len(panels.control_x_coords)))
        Ct2 = np.zeros((len(panels.control_x_coords),len(panels.control_x_coords)))
        for i in range(len(panels.control_x_coords)):
            xi = panels.control_x_coords[i]
            yi = panels.control_y_coords[i]
            theta_i = panels.theta[i]
                    
            for j in range(len(panels.control_x_coords)):
                theta_j = panels.theta[j]
                Sj = panels.lengths[j]
                Xj = panels.x_coords[j]
                Yj = panels.y_coords[j]
                
                if i==j:
                    Cn2[i,j] = 1.0
                    Cn1[i,j] = -1.0
                    Ct2[i,j] = np.pi/2.0
                    Ct1[i,j] = np.pi/2.0
                    
                else:
                    A = -(xi - Xj)*np.cos(theta_j) - (yi - Yj)*np.sin(theta_j)
                    B = (xi - Xj)**2.0 + (yi - Yj)**2.0
                    C = np.sin(theta_i - theta_j)
                    D = np.cos(theta_i - theta_j)
                    E = (xi - Xj)*np.sin(theta_j) - (yi - Yj)*np.cos(theta_j)
                    F = np.log(1.0 + (Sj**2.0 + 2.0*A*Sj)/B)
                    G = np.arctan2(E*Sj, (B + A*Sj))
                    P = (xi - Xj)*np.sin(theta_i - 2.0*theta_j) + (yi - Yj)*np.cos(theta_i - 2.0*theta_j)
                    Q = (xi - Xj)*np.cos(theta_i - 2.0*theta_j) + (yi - Yj)*np.sin(theta_i - 2.0*theta_j)
                    
                    Cn2[i,j] = D+0.5*Q*F/Sj-(A*C+D*E)*G/Sj
                    Cn1[i,j] = 0.5*D*F+C*G-Cn2[i,j]
                    Ct2[i,j] = C+0.5*P*F/Sj+(A*D-C*E)*G/Sj
                    Ct1[i,j] = 0.5*C*F-D*G-Ct2[i,j]
         
        aerodynamic_matrix = np.zeros((len(panels.x_coords),len(panels.x_coords)))
        tangential_matrix = np.zeros((len(panels.x_coords)-1,len(panels.x_coords)))
        for i in range(len(panels.x_coords)):
            for j in range(len(panels.x_coords)):

                if j == 0 and i != panels.n_panels:
                    aerodynamic_matrix[i,j] = Cn1[i,j]
                    tangential_matrix[i,j] = Ct1[i,j]
                elif j > 0 and j < panels.n_panels and i != panels.n_panels:
                    aerodynamic_matrix[i,j] = Cn1[i,j] + Cn2[i,j-1]
                    tangential_matrix[i,j] = Ct1[i,j] + Ct2[i,j-1]
                elif j == panels.n_panels and i != panels.n_panels:
                    aerodynamic_matrix[i,j] = Cn2[i,j-1]
                    tangential_matrix[i,j] = Ct2[i,j-1]
                elif i == panels.n_panels and (j == 0 or j == panels.n_panels):
                    aerodynamic_matrix[i,j] = 1.0
        
        free_stream_matrix = np.sin(panels.theta - alpha*(np.pi/180.0))
        free_stream_matrix = np.append(free_stream_matrix, 0.0)
        
        gamma_prime = np.linalg.solve(aerodynamic_matrix,free_stream_matrix)
        self.v_panels = np.matmul(tangential_matrix, gamma_prime) + np.cos(panels.theta - alpha*(np.pi/180.0))
        
        return self.v_panels
    
    # Solves the total local velocity at each control point based on vortex source method
    # @param Angle of attack
    # @param Panels object that defines airfoil geometry
    def get_velocity_spvp(self, alpha, panels):

        Iij = np.zeros((len(panels.control_x_coords),len(panels.control_x_coords)))
        Jij = np.zeros((len(panels.control_x_coords),len(panels.control_x_coords)))
        Kij = np.zeros((len(panels.control_x_coords),len(panels.control_x_coords)))
        Lij = np.zeros((len(panels.control_x_coords),len(panels.control_x_coords)))
        for i in range(len(panels.control_x_coords)):
            xi = panels.control_x_coords[i]
            yi = panels.control_y_coords[i]
            theta_i = panels.theta[i]
                    
            for j in range(len(panels.control_x_coords)):
                theta_j = panels.theta[j]
                Sj = panels.lengths[j]
                Xj = panels.x_coords[j]
                Yj = panels.y_coords[j]
                
                A = -(xi-Xj)*np.cos(theta_j)-(yi-Yj)*np.sin(theta_j)
                B = (xi-Xj)**2.0+(yi-Yj)**2.0
                Ci = np.sin(theta_i-theta_j)
                Cj = -np.cos(theta_i-theta_j)
                Ck = -np.cos(theta_i-theta_j)
                Cl = np.sin(theta_j-theta_i)
                Di = -(xi-Xj)*np.sin(theta_i)+(yi-Yj)*np.cos(theta_i)
                Djk = (xi-Xj)*np.cos(theta_i)+(yi-Yj)*np.sin(theta_i)
                Dl = (xi-Xj)*np.sin(theta_i)-(yi-Yj)*np.cos(theta_i)
                if B-A*A >= 0.0:
                    E = np.sqrt(B-A*A)    
                else:
                    E = 0.0
                
                if B == 0.0 or E == 0.0:
                    Iij[i,j] = 0.0
                    Jij[i,j] = 0.0
                    Kij[i,j] = 0.0
                    Lij[i,j] = 0.0
                
                else:
                    Iij[i,j] = (Ci/2.0)*np.log((Sj*Sj+2.0*A*Sj+B)/B)+((Di-A*Ci)/E)*(np.arctan2((Sj+A),E)-np.arctan2(A,E))
                    Jij[i,j] = (Cj/2.0)*np.log((Sj*Sj+2.0*A*Sj+B)/B)+((Djk-A*Cj)/E)*(np.arctan2((Sj+A),E)-np.arctan2(A,E))
                    Kij[i,j] = (Ck/2.0)*np.log((Sj*Sj+2.0*A*Sj+B)/B)+((Djk-A*Ck)/E)*(np.arctan2((Sj+A),E)-np.arctan2(A,E))
                    Lij[i,j] = (Cl/2.0)*np.log((Sj*Sj+2.0*A*Sj+B)/B)+((Dl-A*Cl)/E)*(np.arctan2((Sj+A),E)-np.arctan2(A,E))
         
        aerodynamic_matrix = np.zeros((len(panels.x_coords),len(panels.x_coords)))
        for i in range(len(panels.x_coords)):
            for j in range(len(panels.x_coords)):

                if i == len(panels.x_coords) - 1:
                    
                    if j == len(panels.x_coords) - 1:
                        aerodynamic_matrix[i,j] = -np.sum(Lij[0,:]) - np.sum(Lij[len(panels.x_coords)-2,:]) + 2.0*np.pi
                    
                    else:
                        aerodynamic_matrix[i,j] = Jij[0,j] + Jij[len(panels.x_coords)-2,j]

                elif j == len(panels.x_coords) - 1:
                    aerodynamic_matrix[i,j] = -np.sum(Kij[i,:])
                
                elif i == j:
                    aerodynamic_matrix[i,j] = np.pi
                
                else:
                    aerodynamic_matrix[i,j] = Iij[i,j]
        
        free_stream_matrix = -2.0*np.pi*np.sin(panels.theta - alpha*(np.pi/180.0))
        free_stream_matrix = np.append(free_stream_matrix, -2.0*np.pi*(np.cos(panels.theta[0]-alpha*(np.pi/180.0)) + np.cos(panels.theta[len(panels.x_coords)-2]-alpha*(np.pi/180.0))))
        
        source_vortex_soln = np.linalg.solve(aerodynamic_matrix,free_stream_matrix)
        
        self.v_panels = np.zeros(len(panels.x_coords)-1)
        for i in range(len(panels.x_coords)-2):
            term1 = np.cos(panels.theta[i]-alpha*(np.pi/180.0))
            term2 = 1.0 / (2.0*np.pi) * np.sum(source_vortex_soln[0:-1]*Jij[i,:])
            term3 = source_vortex_soln[-1] / 2.0
            term4 = -(source_vortex_soln[-1] / (2.0*np.pi))*np.sum(Lij[i,:])
            
            self.v_panels[i] = term1 + term2 + term3 + term4
        
        return self.v_panels
    
    # Solves the lift, drag, and moment coefficients
    # @param Angle of attack
    # @param Panels object that defines airfoil geometry
    def get_aerodynamics(self, alpha, panels):
        
        self.alpha = alpha
        self.panels = panels
        
        v_panels = self.get_velocity_spvp(alpha, panels)
        self.cp = 1.0 - v_panels**2.0
        
        Cf = -self.cp * panels.lengths * panels.normal
        Cfnet = np.sum(Cf, axis=1)
        Ca = Cfnet[0]
        Cn = Cfnet[1]
        
        self.Cmc4 = 0.0
        for i in range(len(panels.control_x_coords)):
            ra = panels.control_x_coords[i] - 0.25
            rn = panels.control_y_coords[i]
            dca = Cf[0,i]
            dcn = Cf[1,i]
            self.Cmc4 = self.Cmc4 - (dcn*ra-dca*rn)     
            
        self.Cl = Cn*np.cos(alpha*np.pi/180.0) - Ca*np.sin(alpha*np.pi/180.0)
        self.Cdp = Cn*np.sin(alpha*np.pi/180.0) + Ca*np.cos(alpha*np.pi/180.0)
        
        return self.Cl, self.Cdp, self.Cmc4, self.cp
    
    # Calculates the lift and moment curves of a set of panels
    def get_curves(self, panels, n_points):
        
        alpha_curve = np.linspace(-5, 15, n_points)
        
        A = np.zeros((3,3))
        A[0,0] = len(alpha_curve)
        A[1,0] = sum((np.array(alpha_curve)*(np.pi/180.0)))
        A[2,0] = sum((np.array(alpha_curve)*(np.pi/180.0))**2.0)
        A[0,1] = sum((np.array(alpha_curve)*(np.pi/180.0)))
        A[1,1] = sum((np.array(alpha_curve)*(np.pi/180.0))**2.0)
        A[2,1] = sum((np.array(alpha_curve)*(np.pi/180.0))**3.0)
        A[0,2] = sum((np.array(alpha_curve)*(np.pi/180.0))**2.0)
        A[1,2] = sum((np.array(alpha_curve)*(np.pi/180.0))**3.0)
        A[2,2] = sum((np.array(alpha_curve)*(np.pi/180.0))**4.0)
    
        lift_curve = []
        moment_curve = []
        # min_upper_cp = 0.0
        # min_upper_cp_loc = 0.0
        # min_lower_cp = 0.0
        # min_lower_cp_loc = 0.0
        for j in range(n_points):
            Cl, Cd, Cm_c4, cp = self.get_aerodynamics(alpha_curve[j],panels)
            lift_curve.append(Cl)
            moment_curve.append(Cm_c4)
            # if cp:
            #     min_upper_cp = cp
            # if cp:
            #     min_lower_cp = cp
        
        a = len(alpha_curve)*sum(np.array(alpha_curve)*(np.pi/180.0)*np.array(lift_curve))
        b = sum(np.array(alpha_curve)*(np.pi/180.0))*sum(np.array(lift_curve))
        c = len(alpha_curve)*sum((np.array(alpha_curve)*(np.pi/180.0))**2.0)
        d = sum(np.array(alpha_curve)*(np.pi/180.0))**2.0
        lift_slope = (a-b) / (c-d)
        e = sum(np.array(lift_curve))
        f = lift_slope * sum(np.array(alpha_curve)*(np.pi/180.0))
        g = len(alpha_curve)
        zero_lift_angle = 180.0*(f-e) / (g*lift_slope*np.pi)
        
        B = np.zeros((3))
        B[0] = sum(np.array(moment_curve))
        B[1] = sum(np.array(moment_curve) * np.array(alpha_curve) * (np.pi/180.0))
        B[2] = sum(np.array(moment_curve) * (np.array(alpha_curve) * (np.pi/180.0))**2.0)
        C = np.linalg.solve(A,B)
        
        curve_parameters = np.zeros(5)
        curve_parameters[0] = lift_slope
        curve_parameters[1] = zero_lift_angle
        curve_parameters[2] = C[0]
        curve_parameters[3] = C[1]
        curve_parameters[4] = C[2]
        
        return curve_parameters, alpha_curve, lift_curve, moment_curve
        
    # Draws the lift and moment curves
    def draw_curves(self, path, panels, name='', estimated_performance=[], rebuilt_panels=0.0):
        
        _, alpha_curve, lift_curve, moment_curve = self.get_curves(panels, 50)
        
        plot_rebuilt = False
        if isinstance(rebuilt_panels, Panels):
            _, _, rebuilt_lift_curve, rebuilt_moment_curve = self.get_curves(rebuilt_panels, 50)
            plot_rebuilt = True
            
        plot_estimated = False
        if len(estimated_performance)==5:
            estimated_lift_curve = estimated_performance[0] * (alpha_curve*(np.pi/180.0) - estimated_performance[1]*(np.pi/180.0))
            estimated_moment_curve = estimated_performance[2] + estimated_performance[3]*(alpha_curve*(np.pi/180.0)) + estimated_performance[4]*(alpha_curve*(np.pi/180.0))**2.0
            plot_estimated = True
        
        if not os.path.isdir(path):
            os.mkdir(path)
        
        num = 0
        done = False
        while not done:
            if not plot_rebuilt and not plot_estimated:
                done = not (os.path.exists(path + "/lift_" + str(num) + ".png"))
            elif not plot_rebuilt and plot_estimated:
                done = not (os.path.exists(path + "/estimated_lift_" + str(num) + ".png"))
            elif plot_rebuilt and not plot_estimated:
                done = not (os.path.exists(path + "/rebuilt_lift_" + str(num) + ".png"))
            elif plot_rebuilt and plot_estimated:
                done = not (os.path.exists(path + "/estimated_lift_" + str(num) + ".png"))
                done = done and not (os.path.exists(path + "/rebuilt_lift_" + str(num) + ".png"))
            num = num + 1
        num = num - 1
        
        if not plot_rebuilt and not plot_estimated:
            lift_path = path + "/lift_" + str(num) + ".png"
        if plot_estimated:
            lift_path_estimated = path + "/estimated_lift_" + str(num) + ".png"
        if plot_rebuilt:
            lift_path_rebuilt = path + "/rebuilt_lift_" + str(num) + ".png"
        
        if not plot_rebuilt and not plot_estimated:
            plt.close()
            plt.axhline(0.0, color='k', lw=0.75)
            plt.axvline(0.0, color='k', lw=0.75)
            plt.plot(alpha_curve, lift_curve, c='b', lw=2.5)
            plt.xlabel("Angle of Attack [deg]", fontsize="x-large")
            plt.ylabel(r'$C_{l}$'+' [-]', fontsize="x-large")
            if name != '':
                plt.title("Lift Curve for "+name, fontsize="xx-large")
            else:
                plt.title("Lift Curve", fontsize="xx-large")
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            plt.gcf().set_size_inches(8,5.6)
            plt.savefig(lift_path, dpi=200)
            
        if plot_estimated:
            plt.close()
            plt.axhline(0.0, color='k', lw=0.75)
            plt.axvline(0.0, color='k', lw=0.75)
            plt.plot(alpha_curve, lift_curve, c='b', lw=2.5, label='Original')
            plt.plot(alpha_curve, estimated_lift_curve, c='r', lw=2.5, label='Estimated', ls='--')
            plt.xlabel("Angle of Attack [deg]", fontsize="x-large")
            plt.ylabel(r'$C_{l}$'+' [-]', fontsize="x-large")
            if name != '':
                plt.title("Lift Curve for "+name, fontsize="xx-large")
            else:
                plt.title("Lift Curve", fontsize="xx-large")
            plt.legend(fontsize='x-large')
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            plt.gcf().set_size_inches(8,5.6)
            plt.savefig(lift_path_estimated, dpi=200)
            
        if plot_rebuilt:
            plt.close()
            plt.axhline(0.0, color='k', lw=0.75)
            plt.axvline(0.0, color='k', lw=0.75)
            plt.plot(alpha_curve, lift_curve, c='b', lw=2.5, label='Original')
            plt.plot(alpha_curve, rebuilt_lift_curve, c='r', lw=2.5, label='Rebuilt', ls='--')
            plt.xlabel("Angle of Attack [deg]", fontsize="x-large")
            plt.ylabel(r'$C_{l}$'+' [-]', fontsize="x-large")
            if name != '':
                plt.title("Lift Curve for "+name, fontsize="xx-large")
            else:
                plt.title("Lift Curve", fontsize="xx-large")
            plt.legend(fontsize='x-large')
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            plt.gcf().set_size_inches(8,5.6)
            plt.savefig(lift_path_rebuilt, dpi=200)
        
        if not plot_rebuilt and not plot_estimated:
            moment_path = path + "/moment_" + str(num) + ".png"
        if plot_estimated:
            moment_path_estimated = path + "/estimated_moment_" + str(num) + ".png"
        if plot_rebuilt:
            moment_path_rebuilt = path + "/rebuilt_moment_" + str(num) + ".png"

        if not plot_rebuilt and not plot_estimated:
            plt.close()
            plt.axhline(0.0, color='k', lw=0.75)
            plt.axvline(0.0, color='k', lw=0.75)
            plt.plot(alpha_curve, moment_curve, c='b', lw=2.5)
            plt.xlabel("Angle of Attack [deg]", fontsize="x-large")
            plt.ylabel(r'$C_{m}$'+' [-]', fontsize="x-large")
            if name != '':
                plt.title("Moment Curve for "+name, fontsize="xx-large")
            else:
                plt.title("Moment Curve", fontsize="xx-large")
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            plt.gcf().set_size_inches(8,5.6)
            plt.savefig(moment_path, dpi=200)
            
        if plot_estimated:
            plt.close()
            plt.axhline(0.0, color='k', lw=0.75)
            plt.axvline(0.0, color='k', lw=0.75)
            plt.plot(alpha_curve, moment_curve, c='b', lw=2.5, label='Original')
            plt.plot(alpha_curve, estimated_moment_curve, c='r', lw=2.5, label='Estimated', ls='--')
            plt.xlabel("Angle of Attack [deg]", fontsize="x-large")
            plt.ylabel(r'$C_{m}$'+' [-]', fontsize="x-large")
            if name != '':
                plt.title("Moment Curve for "+name, fontsize="xx-large")
            else:
                plt.title("Moment Curve", fontsize="xx-large")
            plt.legend(fontsize='x-large')
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            plt.gcf().set_size_inches(8,5.6)
            plt.savefig(moment_path_estimated, dpi=200)
            
        if plot_rebuilt:
            plt.close()
            plt.axhline(0.0, color='k', lw=0.75)
            plt.axvline(0.0, color='k', lw=0.75)
            plt.plot(alpha_curve, moment_curve, c='b', lw=2.5, label='Original')
            plt.plot(alpha_curve, rebuilt_moment_curve, c='r', lw=2.5, label='Rebuilt', ls='--')
            plt.xlabel("Angle of Attack [deg]", fontsize="x-large")
            plt.ylabel(r'$C_{m}$'+' [-]', fontsize="x-large")
            if name != '':
                plt.title("Moment Curve for "+name, fontsize="xx-large")
            else:
                plt.title("Moment Curve", fontsize="xx-large")
            plt.legend(fontsize='x-large')
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            plt.gcf().set_size_inches(8,5.6)
            plt.savefig(moment_path_rebuilt, dpi=200)