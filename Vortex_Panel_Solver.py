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
    
    def __init__(self, n_panels, panel_spacing="uniform", NACA='random'):
        
        self.n_panels = self.get_n_panels(n_panels)
        self.spacing_code = self.encode_spacing(panel_spacing)
        self.NACA_code = self.get_NACA_code(NACA)
        
        self.x_coords, self.y_coords, self.camber_line, self.NACA_name = self.get_coords(self.n_panels, self.spacing_code, self.NACA_code)
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
    
    # Gets the NACA code associated with input parameters
    # @param input NACA code
    def get_NACA_code(self, NACA):
        
        if NACA.lower() == "rand" or NACA.lower() == "random" or NACA.lower() == "":
            return ""
        
        elif len(NACA)==4 and NACA[0].isdigit() and NACA[1].isdigit() and NACA[2:].isdigit():
            return NACA
        
        else:
            raise Exception("Unrecognized NACA code.")
        
    # Gets the x/c and y/c normalized coordinates of the panels
    # @param Number of panels
    # @param Panel spacing code
    # @param NACA code (if any) to generate
    def get_coords(self, n_panels, spacing_code, NACA_code):
        
        x_coords = self.get_x_coords(n_panels, spacing_code)
        x_coords, y_coords, camber_line, NACA_name = self.get_y_coords(x_coords, NACA_code)
        return x_coords, y_coords, camber_line, NACA_name
    
    # Gets the x/c normalized coordinates of the panels
    # @param Number of panels
    # @param Panel spacing code
    def get_x_coords(self, n_panels, spacing_code):
        
        if spacing_code == 0:
            top_coords = np.linspace(0.0,1.0,(n_panels//2)+1)
            bot_coords = np.linspace(1.0,0.0,(n_panels//2)+1)
            x_coords = np.concatenate((bot_coords, top_coords[1:]))
            return x_coords
        
        elif spacing_code == 1:
            n = (n_panels//2)
            j = np.arange(n+1)
            top_coords = 0.5 - 0.5*np.cos(j*np.pi/n)
            bot_coords = 0.5 + 0.5*np.cos(j*np.pi/n)
            x_coords = np.concatenate((bot_coords, top_coords[1:]))
            return x_coords
        
        else:
            raise Exception("Unrecognized panel spacing code.")

    # Gets the y/c normalized coordinates of the panels and camber updated x/c normalized coords of the panels
    # @param X cooridinates of panels
    # @param NACA code (if any) to generate
    def get_y_coords(self, x_coords, NACA_code):
        
        x_on_c = x_coords[0:len(x_coords)//2+1]
        
        if len(NACA_code) == 0:
            max_thickness = 0.15 * np.random.rand() + 0.10
            max_camber = 0.069 * np.random.rand() + 0.001
            max_camber_loc = 0.4 * np.random.rand() + 0.1
        
        else:
            max_thickness = float(NACA_code[2:])/100.0
            max_camber = float(NACA_code[0]) / 100.0
            max_camber_loc = float(NACA_code[1]) / 10.0
        
        half_thickness = 5.0*max_thickness*(0.2969*x_on_c**0.5-0.1260*x_on_c-0.3516*x_on_c**2.0+0.2843*x_on_c**3.0-0.1015*x_on_c**4.0)
        
        if max_camber == 0.0 or max_camber_loc == 0.0:
            camber_line = np.zeros(len(x_on_c))
            y_upper = half_thickness
            y_lower = -half_thickness
            
        else:
            LE_camber_line = (max_camber * x_on_c / (max_camber_loc**2.0) * (2.0 * max_camber_loc - x_on_c)) * (x_on_c<=max_camber_loc)
            TE_camber_line = (max_camber * (1.0-x_on_c) / (1.0-max_camber_loc)**2.0 * (1.0 + x_on_c - 2.0 * max_camber_loc)) * (x_on_c>max_camber_loc)
            camber_line = LE_camber_line + TE_camber_line
        
            LE_theta = np.arctan((-2.0*max_camber / (max_camber_loc**2.0) * (x_on_c - max_camber_loc))) * (x_on_c<=max_camber_loc)
            TE_theta = np.arctan((-2.0*max_camber / (max_camber_loc-1.0)**2.0 * (x_on_c - max_camber_loc))) * (x_on_c>max_camber_loc)
            theta = LE_theta+TE_theta
        
            y_upper = camber_line + half_thickness * np.cos(theta)
            y_lower = camber_line - half_thickness * np.cos(theta)
            
        y_coords = np.concatenate((y_lower, np.flip(y_upper)[1:]))
        y_coords[0] = 0.0
        y_coords[-1] = 0.0
        
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
        
        plt.close()
        
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
        self.panels=0.0
        self.alpha=0.0
        self.v_panels=0.0
        self.cp=0.0
        self.Cl=0.0
        self.Cdp=0.0
        self.Cmc4=0.0
    
    # Solves the total local velocity at each control point
    # @param Angle of attack
    # @param Panels object that defines airfoil geometry
    def solve_vel(self, alpha, panels):

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
    
    # Solves the lift, drag, and moment coefficients
    # @param Angle of attack
    # @param Panels object that defines airfoil geometry
    def solve(self, alpha, panels):
        
        self.alpha = alpha
        self.panels = panels
        
        v_panels = self.solve_vel(alpha, panels)
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
        
        return self.Cl, self.Cdp, self.Cmc4
    
    def draw(self, path):
        
        if not os.path.isdir(path):
            os.mkdir(path)
        
        num = 0
        done = False
        while not done:
            done = not (os.path.exists(path + "/cp_" + str(num) + ".png"))
            num = num + 1
        num = num - 1
        path = path + "/cp_" + str(num) + ".png"
        
        plt.close()
        
        plt.plot(self.panels.control_x_coords, self.cp, c='b', lw=2.5)
        
        plt.xlabel("X/C [-]", fontsize="x-large")
        plt.ylabel("Y/C [-]", fontsize="x-large")
        plt.title(self.panels.NACA_name+" Pressure Distribution at α=" + str(self.alpha), fontsize="xx-large")
        
        plt.xlim([-0.05, 1.05])
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.gca().invert_yaxis()
                
        y_min = min(self.cp)
        y_range = 1.0 - min(self.cp)
        
        plt.text(0.75, y_min+0.05*y_range, 'Cl = '+str(round(self.Cl, 3)), fontsize='x-large')
        plt.text(0.75, y_min+0.1*y_range, 'Cdp = '+str(round(self.Cdp, 3)), fontsize='x-large')
        plt.text(0.75, y_min+0.15*y_range, 'Cmc/4 = '+str(round(self.Cmc4, 3)), fontsize='x-large')
        
        plt.gcf().set_size_inches(8,5.6)
        plt.savefig(path, dpi=200)

if __name__ == "__main__":
    
    panels = Panels(40, panel_spacing='chebyshev')
    panels.draw("validation")
    
    solver = Solver()
    Cl, Cd, Cm_c4 = solver.solve(round(np.random.rand()*15-5, 1),panels)
    solver.draw("validation")
    