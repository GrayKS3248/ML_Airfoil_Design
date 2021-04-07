# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:43:28 2021

@author: Grayson Schaer
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class Panels:
    
    def __init__(self, n_panels, panel_spacing="chebyshev"):
        
        # Set panel parameters
        self.n_panels = self.get_n_panels(n_panels)
        self.spacing_code = self.encode_spacing(panel_spacing)
        
        # Get panel coordinates
        self.x_coords, self.y_coords, self.camber_line = self.get_coords(self.n_panels, self.spacing_code)
        
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
        x_coords, y_coords, camber_line = self.get_y_coords(x_coords)
        return x_coords, y_coords, camber_line
    
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
        
        max_thickness = 0.25*(np.random.rand()+0.1)
        half_thickness = 5.0*max_thickness*(0.2969*x_on_c**0.5-0.1260*x_on_c-0.3516*x_on_c**2.0+0.2843*x_on_c**3.0-0.1015*x_on_c**4.0)
        
        max_camber = 0.04*(np.random.rand()+0.001)
        max_camber_loc = 0.5*(np.random.rand()+0.1)
        LE_camber_line = (max_camber * x_on_c / (max_camber_loc**2.0) * (2.0 * max_camber_loc - x_on_c)) * (x_on_c<=max_camber_loc)
        TE_camber_line = (max_camber * (1.0-x_on_c) / (1.0-max_camber_loc)**2.0 * (1.0 + x_on_c - 2.0 * max_camber_loc)) * (x_on_c>max_camber_loc)
        camber_line = LE_camber_line + TE_camber_line
        
        LE_theta = np.arctan((-2.0*max_camber / (max_camber_loc**2.0) * (x_on_c - max_camber_loc))) * (x_on_c<=max_camber_loc)
        TE_theta = np.arctan((-2.0*max_camber / (max_camber_loc-1.0)**2.0 * (x_on_c - max_camber_loc))) * (x_on_c>max_camber_loc)
        theta = LE_theta+TE_theta
        
        y_upper = camber_line + half_thickness * np.cos(theta)
        y_lower = camber_line - half_thickness * np.cos(theta)
        y_coords = np.concatenate((y_upper, np.flip(y_lower)[1:]))
        y_coords[0] = 0.00025
        y_coords[-1] = -0.00025
        
        x_upper = x_on_c - half_thickness * np.sin(theta)
        x_lower = x_on_c + half_thickness * np.sin(theta)
        x_coords = np.concatenate((x_upper, np.flip(x_lower)[1:]))
        x_coords[x_coords < 0.0] = 0.0
        x_coords[x_coords > 1.0] = 1.0
        
        return x_coords, y_coords, camber_line
    
    # Renders and save the panels
    # @param save path
    def draw_panels(self, path):

        if not os.path.isdir(path):
            os.mkdir(path)
        
        num = 0
        done = False
        while not done:
            done = not (os.path.exists(path + "/airfoil_" + str(num) + ".png"))
            num = num + 1
        num = num - 1
        path = path + "/airfoil_" + str(num) + ".png"
        
        plt.plot(self.x_coords, self.y_coords, 'o', color='k')
        plt.plot(self.x_coords, self.y_coords, color='b', lw=2.0)
        plt.plot(self.x_coords[0:len(self.x_coords)//2+1], self.camber_line, lw=2.0, ls="--", c='r')
        plt.axhline(0.0, lw=2.0, c='r')
        
        plt.xlabel("X/C [-]", fontsize="large")
        plt.ylabel("Y/C [-]", fontsize="large")
        plt.title("Airfoil " + str(num), fontsize="xx-large")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([-0.15, 0.20])
        
        plt.savefig(path, dpi=200)