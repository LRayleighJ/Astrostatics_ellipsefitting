import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from ellipse_fit import fitting as eft

# NGC7597 349.62597 18.68925

# decide which band to use and define the center and size you want to cut.
bandlist = ["g","i","r","u","z"] # use i band
halfwidth_cut = 50
center_cut = (472, 1614)

# load data
filename_image = "./data/correctframe/frame-i-006366-4-0132.fits"
hdul = fits.open(filename_image)
fitdata = hdul[0].data

# SDSS field center: (472, 1614)
fitdata_cut = fitdata[center_cut[0]-halfwidth_cut:center_cut[0]+halfwidth_cut,
                      center_cut[1]-halfwidth_cut:center_cut[1]+halfwidth_cut]

# fit the ellipse, and get the eccentricity of the ellipse and the inclination of the main axis
## present the pixel luminosity distribution
flatfitdata = fitdata_cut.flatten()

plt.figure()
plt.hist(flatfitdata,bins=100,range=[0.,5.])
plt.savefig("./figs/luminosity_distribution.pdf")

lumi_border_list = np.linspace(0.1,2,20)
lumi_center_list = np.linspace(0.15,1.95,19)

ecc_list = []
alpha_list = []
A_list = []
B_list = []
lumi_list = []

# plot the fitting ellipse
# fig, ax = plt.subplots() 
for lumi_index in range(len(lumi_center_list)):
    try:
        x_samples, y_samples = eft.get_contour_index(lumi_border_list[lumi_index], lumi_border_list[lumi_index+1], fitdata_cut)
        ellipse_center = np.array([np.mean(x_samples),np.mean(y_samples)])
        # Normalize
        x_samples = x_samples-np.mean(x_samples)
        y_samples = y_samples-np.mean(y_samples)

        # fit

        a_res, b_res, alpha_res = eft.fit_ellipse(x_samples, y_samples,init_params=(-10,-2,10,2,15)) #plot_axis=ax,xyadd=ellipse_center,  long axis, short axis, inclination of the main axis (radius, not degree)

        ecc = np.sqrt(1-b_res**2/a_res**2)
        ecc_list.append(ecc)
        alpha_list.append(alpha_res)
        A_list.append(a_res)
        B_list.append(b_res)
        lumi_list.append(lumi_center_list[lumi_index])
    except:
        print("fitting failed at lumi=%.2f"%lumi_center_list[lumi_index])
        continue

#plt.savefig("test_ellipse_profile.pdf")
#plt.close()

# test the static values
plt.figure()
plt.subplot(311)
plt.scatter(lumi_list,A_list,s=2)
plt.ylabel("A/pix")
plt.subplot(312)
plt.scatter(lumi_list,ecc_list,s=2)
plt.ylabel("eccentricity")
plt.subplot(313)
plt.scatter(lumi_list,alpha_list,s=2)
plt.xlabel("luminosity")
plt.ylabel("inclination/rad")
plt.suptitle("NGC7597")
plt.savefig("test_regression.pdf")
plt.close()

print("eccentricity and inclination(deg): ",np.mean(ecc_list),np.mean(alpha_list)*180/np.pi)

# fit the sersic function CAUTION: JUST a TEST, DO NOT use
'''
p0 = [1, 1]
Para = leastsq(eft.err_sersic, p0, args=(A_list,lumi_list))
I0, R0 = Para[0]

print("sersic fitting: I0=%.2f, R0=%.4f"%(I0, R0))

fit_lumi = eft.sersic(Para[0],np.array(A_list))

plt.figure()
plt.scatter(A_list,lumi_list,s=10,marker="x",c="blue")
plt.plot(A_list,fit_lumi,ls="--",label="sersic: I0=%.2f, R0=%.4f"%(I0, R0),c="g")
plt.xlabel("A/pix")
plt.ylabel("luminosity")
plt.legend()
plt.savefig("test_sersic.pdf")
'''
## correct fit
ecc = np.mean(ecc_list)
alpha = np.mean(alpha_list)
A_axis = np.linspace(5,20,5)

x_line = np.linspace(-20,20,100)
y_line = x_line*np.tan(np.pi/2-alpha)

print(fitdata_cut.shape)
X_axis = np.array(range(fitdata_cut.shape[0]))
Y_axis = np.array(range(fitdata_cut.shape[1]))
X_axis = X_axis-np.mean(X_axis)
Y_axis = Y_axis-np.mean(Y_axis)
X_grid,Y_grid = np.meshgrid(X_axis,Y_axis)

fig,ax = plt.subplots()
# ax.plot(x_line,y_line)
for A in A_axis:
    eft.draw_ellipse(ax, center=[0,0], A=A, e=ecc, alpha=np.pi/2-alpha)
axc = ax.contourf(X_grid, Y_grid, fitdata_cut)
ax.axis("scaled")
fig.colorbar(axc)
plt.savefig("test_ellipse_profile.pdf")

# fit sersic
X_rotate_grid = X_grid*np.cos(np.pi/2-alpha)+Y_grid*np.sin(np.pi/2-alpha)
Y_rotate_grid = -X_grid*np.sin(np.pi/2-alpha)+Y_grid*np.cos(np.pi/2-alpha)
A_grid = np.sqrt(X_rotate_grid**2+Y_rotate_grid**2/(1-ecc**2))

A_list = np.linspace(4,20,50)
A_center_list = []
I_list = []
for A_index in range(len(A_list)-1):
    x_samples, y_samples = eft.get_contour_index(A_list[A_index], A_list[A_index+1], A_grid)
    I_value_list = []
    value_check_list = []
    for index in range(len(x_samples)):
        I_value_list.append(fitdata_cut[x_samples[index]][y_samples[index]])
        value_check_list.append(A_grid[x_samples[index]][y_samples[index]])
    if len(I_value_list)==0:
        continue
    I_list.append(np.mean(I_value_list))
    A_center_list.append((A_list[A_index]+A_list[A_index+1])/2)


p0 = [1, 1]
Para = leastsq(eft.err_sersic, p0, args=(A_center_list,I_list))
I0, R0 = Para[0]

print("sersic fitting: I0=%.2f, R0=%.4f"%(I0, R0))

fit_lumi = eft.sersic(Para[0],np.array(A_center_list))

plt.figure()
plt.scatter(A_center_list,I_list,s=10,marker="x",c="blue")
plt.plot(A_center_list,fit_lumi,ls="--",label="sersic: I0=%.2f, R0=%.4f"%(I0, R0),c="g")
plt.xlabel("A/pix")
plt.ylabel("luminosity")
plt.legend()
plt.savefig("test_fit_sersic.pdf")



