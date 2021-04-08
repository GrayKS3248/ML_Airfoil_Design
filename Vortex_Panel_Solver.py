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
    
    def __init__(self, n_panels, panel_spacing="chebyshev"):
        
        self.n_panels = self.get_n_panels(n_panels)
        self.spacing_code = self.encode_spacing(panel_spacing)
        
        self.x_coords, self.y_coords, self.camber_line, self.NACA_name = self.get_coords(self.n_panels, self.spacing_code)
        self.control_x_coords, self.control_y_coords = self.get_points(self.x_coords, self.y_coords)
        self.normal = self.get_normal(self.x_coords, self.y_coords)
        self.lengths = self.get_length(self.x_coords, self.y_coords)
        self.theta = self.get_angles(self.x_coords, self.y_coords, self.lengths)
        
    # Ensures the passed number of panels is valid
    # @param Number of panels to create
    def get_n_panels(self, n_panels):
        
        if int(round(n_panels)) % 2 == 0:
            return int(round(n_panels))
        
        else:
            raise Exception("Invalid number of panels (must be even).")
        
    # Gets the panel spacing code based on input string
    # @param Panel spacing input string
    def encode_spacing(self, panel_spacing):
        
        if panel_spacing.lower() == "uniform":
            return 0
        
        elif panel_spacing.lower() == "chebyshev":
            return 1
        
        else:
            raise Exception("Invalid panel spacing selection.")
    
    # Gets the x/c and y/c normalized coordinates of the panels
    # @param Number of panels
    # @param Panel spacing code
    def get_coords(self, n_panels, spacing_code):
        
        x_coords = self.get_x_coords(n_panels, spacing_code)
        x_coords, y_coords, camber_line, NACA_name = self.get_y_coords(x_coords)
        return x_coords, y_coords, camber_line, NACA_name
    
    # Gets the x/c normalized coordinates of the panels
    # @param Number of panels
    # @param Panel spacing code
    def get_x_coords(self, n_panels, spacing_code):
        
        if spacing_code == 0:
            top_coords = np.linspace(1.0,0.0,(n_panels//2)+1)
            bot_coords = np.linspace(0.0,1.0,(n_panels//2)+1)
            x_coords = np.concatenate((top_coords, bot_coords[1:]))
            return x_coords
        
        elif spacing_code == 1:
            n = (n_panels//2)
            j = np.arange(n+1)
            top_coords = 0.5 + 0.5*np.cos(j*np.pi/n)
            bot_coords = 0.5 - 0.5*np.cos(j*np.pi/n)
            x_coords = np.concatenate((top_coords, bot_coords[1:]))
            return x_coords
        
        else:
            raise Exception("Unrecognized panel spacing code.")

    # Gets the y/c normalized coordinates of the panels and camber updated x/c normalized coords of the panels
    # @param X cooridinates of panels
    def get_y_coords(self, x_coords):
        
        x_on_c = x_coords[0:len(x_coords)//2+1]
        
        max_thickness = 0.15 * np.random.rand() + 0.10
        half_thickness = 5.0*max_thickness*(0.2969*x_on_c**0.5-0.1260*x_on_c-0.3516*x_on_c**2.0+0.2843*x_on_c**3.0-0.1015*x_on_c**4.0)
        
        max_camber = 0.069 * np.random.rand() + 0.001
        max_camber_loc = 0.4 * np.random.rand() + 0.1
        LE_camber_line = (max_camber * x_on_c / (max_camber_loc**2.0) * (2.0 * max_camber_loc - x_on_c)) * (x_on_c<=max_camber_loc)
        TE_camber_line = (max_camber * (1.0-x_on_c) / (1.0-max_camber_loc)**2.0 * (1.0 + x_on_c - 2.0 * max_camber_loc)) * (x_on_c>max_camber_loc)
        camber_line = LE_camber_line + TE_camber_line
        
        LE_theta = np.arctan((-2.0*max_camber / (max_camber_loc**2.0) * (x_on_c - max_camber_loc))) * (x_on_c<=max_camber_loc)
        TE_theta = np.arctan((-2.0*max_camber / (max_camber_loc-1.0)**2.0 * (x_on_c - max_camber_loc))) * (x_on_c>max_camber_loc)
        theta = LE_theta+TE_theta
        
        y_upper = camber_line + half_thickness * np.cos(theta)
        y_lower = camber_line - half_thickness * np.cos(theta)
        y_coords = np.concatenate((y_upper, np.flip(y_lower)[1:]))
        
        y_coords[0] = 0.001
        y_coords[-1] = -0.001
        x_coords[0] = 1.0
        x_coords[-1] = 1.0
        
        NACA_name = "NACA_" + str(round(max_camber*100.0)) + str(round(max_camber_loc*10.0)) + str(round(max_thickness*100.0))
        
        return x_coords, y_coords, camber_line, NACA_name
    
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
        rotation = np.array([[0.0, 1.0],[-1.0, 0.0]])
       
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
        
        theta = np.arctan(np.diff(y_coords)/np.diff(x_coords))
        return theta
    
    # Renders and save the panels
    # @param save path
    def draw(self, path):

        if not os.path.isdir(path):
            os.mkdir(path)
        
        num = 0
        done = False
        while not done:
            done = not (os.path.exists(path + "/airfoil_" + str(num) + ".png"))
            num = num + 1
        num = num - 1
        path = path + "/airfoil_" + str(num) + ".png"
        
        normal_x_coords_start = panels.x_coords[1:]-0.5*np.diff(panels.x_coords)
        normal_y_coords_start = panels.y_coords[1:]-0.5*np.diff(panels.y_coords)
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
        plt.title(self.NACA_name, fontsize="xx-large")
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.15, 0.20])
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize='large')
        plt.yticks([-0.15, -0.05, 0.05, 0.15, 0.25],fontsize='large')
        
        plt.gcf().set_size_inches(8,2.8)
        plt.savefig(path, dpi=200)
        
class Solver:
    
    def __init__(self):
        pass
    
    # Solves the total local velocity at each control point
    # @param Angle of attack
    # @param Panels object that defines airfoil geometry
    def solve_local_vel(self, alpha, panels):
        
        
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
                    G = np.arctan(E*Sj / (B + A*Sj))
                    P = (xi - Xj)*np.sin(theta_i - 2.0*theta_j) + (yi - Yj)*np.cos(theta_i - 2.0*theta_j)
                    Q = (xi - Xj)*np.cos(theta_i - 2.0*theta_j) + (yi - Yj)*np.sin(theta_i - 2.0*theta_j)
                    
                    Cn2[i,j] = D+0.5*Q*F/Sj-(A*C+D*E)*G/Sj
                    Cn1[i,j] = 0.5*D*F+C*G-Cn2[i,j]
                    Ct2[i,j] = C+0.5*P*F/Sj+(A*D-C*E)*G/Sj
                    Ct1[i,j] = 0.5*C*F-D*G-Ct2[i,j]
         
        aerodynamic_matrix = np.zeros((len(panels.x_coords),len(panels.x_coords)))
        normal_matrix = np.zeros((len(panels.x_coords)-1,len(panels.x_coords)))
        tangential_matrix = np.zeros((len(panels.x_coords)-1,len(panels.x_coords)))
        for i in range(len(panels.x_coords)):
            for j in range(len(panels.x_coords)):

                if j == 0 and i != panels.n_panels:
                    aerodynamic_matrix[i,j] = Cn1[i,j]
                    normal_matrix[i,j] = Cn1[i,j]
                    tangential_matrix[i,j] = Ct1[i,j]
                elif j > 0 and j < panels.n_panels and i != panels.n_panels:
                    aerodynamic_matrix[i,j] = Cn1[i,j] + Cn2[i,j-1]
                    normal_matrix[i,j] = Cn1[i,j] + Cn2[i,j-1]
                    tangential_matrix[i,j] = Ct1[i,j] + Ct2[i,j-1]
                elif j == panels.n_panels and i != panels.n_panels:
                    aerodynamic_matrix[i,j] = Cn2[i,j-1]
                    normal_matrix[i,j] = Cn2[i,j-1]
                    tangential_matrix[i,j] = Ct2[i,j-1]
                elif i == panels.n_panels and (j == 0 or j == panels.n_panels):
                    aerodynamic_matrix[i,j] = 1.0
        
        free_stream_matrix = np.sin(panels.theta - alpha*(np.pi/180.0))
        free_stream_matrix = np.append(free_stream_matrix, 0.0)
        
        gamma_prime = np.linalg.solve(aerodynamic_matrix,free_stream_matrix)
        vt_panels = np.matmul(tangential_matrix, gamma_prime) + np.cos(panels.theta - alpha*(np.pi/180.0))
        vn_panels = np.matmul(normal_matrix, gamma_prime) - np.sin(panels.theta - alpha*(np.pi/180.0))
        
        return vt_panels
    
    # Solves the lift, drag, and moment coefficients
    # @param Angle of attack
    # @param Panels object that defines airfoil geometry
    def solve(self, alpha, panels):
        
        v_panels = self.solve_local_vel(alpha, panels)
        cp = 1.0 - v_panels**2.0
        
        cp_upper = np.flip(cp[0:len(cp)//2])
        cp_lower = cp[len(cp)//2:]
        
        control_x_upper = np.flip(panels.control_x_coords[0:len(cp)//2])
        control_x_lower = panels.control_x_coords[len(cp)//2:]
        Cn = np.trapz((cp_lower - cp_upper), x=(control_x_upper+control_x_lower)/2.0)
        
        x_coords_upper = np.flip(panels.x_coords[0:len(cp)//2+1])
        x_coords_lower = panels.x_coords[len(cp)//2:]
        y_coords_upper = np.flip(panels.y_coords[0:len(cp)//2+1])
        y_coords_lower = panels.y_coords[len(cp)//2:]
        dy_dx_upper = np.diff(y_coords_upper)/np.diff(x_coords_upper)
        dy_dx_lower = np.diff(y_coords_lower)/np.diff(x_coords_lower)
        Ca = np.trapz((cp_upper*dy_dx_upper - cp_lower*dy_dx_lower), x=(control_x_upper+control_x_lower)/2.0)
    
        Cm_c4 = np.trapz((cp_upper - cp_lower) * (0.25 - (control_x_upper+control_x_lower)/2.0), x=(control_x_upper+control_x_lower)/2.0)
        Cl = Cn*np.cos(alpha*np.pi/180.0) - Ca*np.sin(alpha*np.pi/180.0)
        Cd = Cn*np.sin(alpha*np.pi/180.0) + Ca*np.cos(alpha*np.pi/180.0)
        
        return Cl, Cd, Cm_c4

if __name__ == "__main__":
    panels = Panels(32, panel_spacing='Chebyshev')
    panels.draw("validation")
    solver = Solver()
    
    Cl, Cd, Cm_c4 = solver.solve(2.0,panels)
    