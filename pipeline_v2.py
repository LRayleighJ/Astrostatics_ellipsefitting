import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from scipy.special import gammaincinv
from scipy.optimize import leastsq
from photutils.isophote import Ellipse
from photutils.isophote import EllipseGeometry
import lmfit
from lmfit import Minimizer, Parameters, report_fit, Parameter, minimize, fit_report ,conf_interval
import corner

# NGC7597 349.62597 18.68925
# ............................................parameters..............................................................
band = "i"# ["g","i","r","u","z"] # use i band
halfwidth_cut = 50 # half of fig size
center_cut = (472, 1614) # center of galaxy
sma_list = [3.,6.,10.,15.,23.,28.]
num_drop_center_points = 10

# ............................................pipeline................................................................
# load data
filename_image = './data/correctframe/frame-%s-006366-4-0132.fits'%(band,)
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
plt.close()

#fitting surface luminosity using photutils
ellipse = Ellipse(fitdata_cut)
isolist = ellipse.fit_image()
with open("isolist_to_table.txt","w") as f:
    print(isolist.to_table(), file=f)

fig, ax = plt.subplots(figsize=(12, 8))
ax1 = ax.imshow(fitdata_cut,cmap=cm.jet, norm=LogNorm())
plt.colorbar(ax1)

for sma in sma_list:
    iso = isolist.get_closest(sma)
    x, y, = iso.sampled_coordinates()
    ax.plot(x, y, color='white',linewidth=3)
    #print('Closest SMA = {:f}'.format(iso.sma))
    # this method on an Isophote instance returns the (x, y) coordinates of
    # the sampled points in the image.
plt.savefig('test_ellipse_profile.pdf')
plt.close()

fig,ax = plt.subplots(figsize=(12, 8))
# ax.scatter(isolist.sma, isolist.intens)
ax.errorbar(isolist.sma, isolist.intens, yerr=isolist.int_err, fmt='o', markersize=2)
ax.set_xlabel('sma')
ax.set_ylabel('luminosity')
ax.set_title("brightness profile")
plt.savefig("test_luminosity_profile.pdf")
plt.close()

# arguments of ellipse fitting 
fig, axes = plt.subplots(2,2,figsize=(12, 8))
ax1 = axes[0,0]
ax1.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err, fmt='o', markersize=4) # eps = (a − b) / a, not eccentricity
ax1.set_xlabel('Semimajor axis length')
ax1.set_ylabel('Ellipticity')

ax1 = axes[0,1]
ax1.errorbar(isolist.sma, isolist.pa, yerr=isolist.pa_err, fmt='o', markersize=4) 
ax1.set_xlabel('Semimajor axis length')
ax1.set_ylabel('position angle/rad')

ax1 = axes[1,0]
ax1.errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o', markersize=4) 
ax1.set_xlabel('Semimajor axis length')
ax1.set_ylabel('center x coordinate/pix')

ax1 = axes[1,1]
ax1.errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o', markersize=4) 
ax1.set_xlabel('Semimajor axis length')
ax1.set_ylabel('center y coordinate/pix')

plt.savefig("test_ellipse_args.pdf")
plt.close()

# fitting 1D sersic
def fit_sersic(params, r, lumi, error):
    v = params.valuesdict()
    model = v['I0'] * np.exp(-(r/v['r0'])**0.25)
    return (model-lumi)/error


def sersic(r, I0, r0):
    return I0*np.exp(-(r/r0)**0.25)


params = Parameters()
params['I0'] = Parameter(name='I0',value=300,min=0.01,max=1000)
params['r0'] = Parameter(name = 'r0',value=10,min=0.000001,max=1)
minner = Minimizer(fit_sersic, params, fcn_args=(isolist.sma[num_drop_center_points:],isolist.intens[num_drop_center_points:],isolist.int_err[num_drop_center_points:]))
fit_sersic_output = minner.minimize(method='leastsq')

I0_fit = fit_sersic_output.params["I0"].value
r0_fit = fit_sersic_output.params["r0"].value

print("Fitting result    I0:%.2f, r0:%.6f"%(I0_fit,r0_fit))

fig,ax = plt.subplots(figsize=(6, 4))
ax.plot(isolist.sma[num_drop_center_points:],sersic(isolist.sma,I0_fit, r0_fit)[num_drop_center_points:])
ax.errorbar(isolist.sma, isolist.intens, yerr=isolist.int_err, fmt='o', markersize=2)
ax.set_xlabel('sma')
ax.set_ylabel('luminosity')
ax.set_title("brightness profile")
plt.savefig("test_luminosity_fitting.pdf")
plt.close()

# Uncertainty analysis of parameters
ci = lmfit.conf_interval(minner, fit_sersic_output)
lmfit.printfuncs.report_ci(ci)

fig, ax = plt.subplots(figsize=(6,6))
cx, cy, grid = lmfit.conf_interval2d(minner, fit_sersic_output, 'I0', 'r0', 30, 30)
ctp0 = ax.contour(cx, cy, grid, np.linspace(0, 1, 11))
ax.set_xlabel('I0')
ax.set_ylabel('r0')
plt.savefig("test_confidence_ellipse.pdf")
plt.close()

# MCMC
res = minner.minimize(method='emcee', burn=400,steps=1000, nwalkers=200, thin=20, is_weighted=True)

#plt.figure(figsize=(12, 8))
emcee_plot = corner.corner(res.flatchain, labels=res.var_names,
truths=list(res.params.valuesdict().values()))
plt.savefig('test_mcmc_sampling.pdf')
plt.close()

print('median of posterior probability distribution')
print('--------------------------------------------')
lmfit.report_fit(res.params)

highest_prob = np.argmax(res.lnprob)
hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
mle_soln = res.chain[hp_loc]
for i, par in enumerate(params):
    params[par].value = mle_soln[i]
print('\nMaximum Likelihood Estimation from emcee ')
print('-------------------------------------------------')
print('Parameter MLE Value Median Value Uncertainty')
fmt = ' {:5s} {:11.5f} {:11.5f} {:11.5f}'.format
for name, param in params.items():
    print(fmt(name, param.value, res.params[name].value, res.params[name].stderr))

print('\nError estimates from emcee:')
print('------------------------------------------------------')
print('Parameter -2sigma -1sigma median +1sigma +2sigma')
for name in params.keys():
    quantiles = np.percentile(res.flatchain[name],
    [2.275, 15.865, 50, 84.135, 97.275])
    median = quantiles[2]
    err_m2 = quantiles[0] - median
    err_m1 = quantiles[1] - median
    err_p1 = quantiles[3] - median
    err_p2 = quantiles[4] - median
    fmt = ' {:8s}{:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
    print(fmt(name, err_m2, err_m1, median, err_p1, err_p2))

# FISHER matrix forecast: Coming SOON