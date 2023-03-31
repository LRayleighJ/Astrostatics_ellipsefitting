import matplotlib.pyplot as plt
import scipy.optimize as so
import numpy as np

def sersic(para, x):
    I0, R0 = para
    return I0*np.exp(-(x/R0)**0.25)
 
def err_sersic(para, x, y):
    return (sersic(para, x) - y)**2

def get_contour_index(x_dn, x_up, data):
    data_judge = (data<x_up)&(data>=x_dn)
    return np.where(data_judge==True)

def get_bin_center(X):
    X_bin = []
    for X_index in range(len(X)-1):
        X_bin.append((X[X_index]+X[X_index+1])/2)
    return np.array(X_bin)

def my_fun(parameters, x_samples, y_samples):
    # The coordinates of the two focal points and the sum of the distances from the points on the ellipse to the two focal points are used as optimization parameters
    x_focus_1,y_focus_1,x_focus_2,y_focus_2,A2 = parameters # A2: 2*a,long axis
    # Calculate the actual distance
    sum_of_actual_distance_between_edge_and_two_focus= ((x_samples- x_focus_1) ** 2 + (y_samples-y_focus_1) ** 2) ** 0.5 + ((x_samples- x_focus_2) ** 2 + (y_samples-y_focus_2) ** 2) ** 0.5

    # print(np.average(sum_of_actual_distance_between_edge_and_two_focus))
    # Return Variance
    return np.sum(((sum_of_actual_distance_between_edge_and_two_focus - A2) ** 2)/(len(x_samples)-1))

def fit_ellipse(x_samples, y_samples,init_params = None,print_fitting_info=False,plot_axis=None,xyadd = None):# 
    if init_params == None:
        raise(RuntimeError,"init_params should not be NONE")
    # Normalization
    '''
    vmax= max(np.max(x_samples), np.max(y_samples))
    x_samples= x_samples / vmax
    y_samples= y_samples / vmax
    '''
    res_optimized = so.minimize(fun=my_fun, x0=init_params, args=(x_samples, y_samples))
    if res_optimized.success:
        if print_fitting_info:
            print(res_optimized)
        x1_res, y1_res, x2_res, y2_res, l2_res = res_optimized.x
        # Generate elliptic curve based on the optimized function
        # Calculate the elliptical declination
        alpha_res= np.arctan((y2_res- y1_res)/(x2_res-x1_res))
        # Calculate the distance between two focal points
        l_ab= ((y2_res- y1_res)**2+ (x2_res-x1_res)**2)**0.5
        # Calculate the length of the long (short) axis
        a_res= l2_res/2
        # Calculate short (long) axis length
        b_res=  ((l2_res/2)**2- (l_ab/2)**2)**0.5

        # Polar axis sequence
        theta_res = np.linspace(0.0, 6.28, 100)
        # Generate points on the ellipse
        x_res = a_res * np.cos(theta_res) * np.cos(alpha_res) \
                - b_res * np.sin(theta_res) * np.sin(alpha_res)
        y_res = b_res * np.sin(theta_res) * np.cos(alpha_res) \
                + a_res * np.cos(theta_res) * np.sin(alpha_res)
        
        
        # plt.style.use("one")
        if plot_axis:
            plot_axis.plot(x_res+xyadd[0], y_res+xyadd[1], color="deepskyblue")
            # plot_axis.scatter(x_samples, y_samples, color="magenta", marker="+", zorder=1, s=80, label="samples")
            
            # plot_axis.scatter(np.array([x1_res,x2_res]), np.array([y1_res,y2_res]),zorder=3, color="r", label= "focus point")
            # plot_axis.xlabel("$x$")
            # plot_axis.ylabel("$y$")
            # plot_axis.legend()
            # vmax = max(np.max(plt.xlim()), np.max(plt.ylim()))
            # vmin = min(np.min(plt.xlim()), np.min(plt.ylim()))
            # plot_axis.ylim([1.1 * vmin - 0.1 * vmax, 1.1 * vmax - 0.1 * vmin])
            # plot_axis.xlim([1.25 * vmin - 0.25 * vmax, 1.25 * vmax - 0.25 * vmin])
            # plt.savefig("./figs/a=%.3f_b=%.3f_theta=%.2fdeg.pdf"%(a_res, b_res, alpha_res))

        return a_res, b_res, alpha_res
    else:
        print("Not success")
        raise(RuntimeError,"Ellipse-fitting is failed")
    
def draw_ellipse(ax, center, A, e, alpha):
    # Polar axis sequence
    theta_res = np.linspace(0.0, 2*np.pi, 100)
    B = np.sqrt(1-e**2)*A
    P = B**2/A
    R = P/(1+e*np.cos(theta_res-alpha))
    # Generate points on the ellipse
    x_res = R*np.cos(theta_res) + e*A*np.cos(alpha)+center[0]
    y_res = R*np.sin(theta_res) + e*A*np.sin(alpha)+center[1]
    ax.plot(x_res, y_res, color="deepskyblue",lw=0.5)