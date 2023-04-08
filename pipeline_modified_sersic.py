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
from ellipse_fit import fitting as eft

# NGC7597 ra=349.62597 dec=18.68925
GALAXY_NAME = "NGC7597"
# ............................................parameters..............................................................
band = "i"# ["g","i","r","u","z"] # use i band
halfwidth_cut = 50 # half of fig size
center_cut = (472, 1614) # center of galaxy
sma_list = [3.,6.,10.,15.,23.,28.]# semimajor axis when plotting the fitting ellipse
num_drop_center_points = 15 # number of center points dropping when fitting sersic function 

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
def sersic(r, I0, r0):
    return I0*np.exp(-(r/r0)**0.25)


def fit_sersic(params, r, lumi, error):
    v = params.valuesdict()
    model = v['I0'] * np.exp(-(r/v['r0'])**0.25)
    return (model-lumi)/error


def bn(n):
    return 2*n-1/3+4/405/n+46/25515/n**2+131/1148175/n**3-2194697/30690717750/n**4


def pbn(n):
    return 2-4/405/n**2-2*46/25515/n**3-3*131/1148175/n**4+4*2194697/30690717750/n**5


def modified_sersic(r,I0,r0,n):
    return I0*np.exp(-bn(n)*((r/r0)**(1/n)-1))


def fit_modified_sersic(params, r, lumi, error):
    v = params.valuesdict()
    model = modified_sersic(r,v["I0"],v["r0"],v["n"])
    return (model-lumi)/error


params = Parameters()
params['I0'] = Parameter(name='I0',value=0.5,min=0.01,max=1000)
params['r0'] = Parameter(name = 'r0',value=30,min=0.000001,max=60)
params['n'] = Parameter(name = 'r0',value=4,min=0.000001,max=6)
minner = Minimizer(fit_modified_sersic, params, fcn_args=(isolist.sma[num_drop_center_points:],isolist.intens[num_drop_center_points:],isolist.int_err[num_drop_center_points:]))
fit_sersic_output = minner.minimize(method='leastsq')

I0_fit = fit_sersic_output.params["I0"].value
r0_fit = fit_sersic_output.params["r0"].value
n_fit  = fit_sersic_output.params["n"].value

print("Fitting result(leastsq)    I0:%.2f, r0:%.6f, n:%2f"%(I0_fit,r0_fit,n_fit))

'''
fig,ax = plt.subplots(figsize=(6, 4))
ax.plot(isolist.sma[num_drop_center_points:],modified_sersic(isolist.sma,I0_fit, r0_fit, n_fit)[num_drop_center_points:])
ax.errorbar(isolist.sma, isolist.intens, yerr=isolist.int_err, fmt='o', markersize=2)
ax.set_xlabel('sma')
ax.set_ylabel('luminosity')
ax.set_title("brightness profile")
plt.savefig("test_luminosity_fitting.pdf")
plt.close()
'''

# leastsq is not accurate
'''
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
'''
# MCMC
res = minner.minimize(method='emcee', burn=500,steps=1000, nwalkers=200, thin=20, is_weighted=True)

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
# using normalized parameters
# I0_fit r0_fit n_fit
I0_fit = res.params["I0"].value
r0_fit = res.params["r0"].value
n_fit = res.params["n"].value

print("args given by emcee(I,r,n): ", I0_fit, r0_fit, n_fit)

sma_fit = isolist.sma[num_drop_center_points:]
lumi_fit = isolist.intens[num_drop_center_points:]
lumi_err_fit = isolist.int_err[num_drop_center_points:]

# calculate Fisher Matrix
pfpi = I0_fit*np.exp(-bn(n_fit)*((sma_fit/r0_fit)**(1/n_fit)-1))
pfpr = I0_fit*np.exp(-bn(n_fit)*((sma_fit/r0_fit)**(1/n_fit)-1))*bn(n_fit)*1/n_fit*sma_fit**(1/n_fit)/r0_fit**(1/n_fit+1)*r0_fit
pfpn = I0_fit*np.exp(-bn(n_fit)*((sma_fit/r0_fit)**(1/n_fit)-1))*(-pbn(n_fit)*((sma_fit/r0_fit)**(1/n_fit)-1)+bn(n_fit)*1/n_fit**2*(sma_fit/r0_fit)**(1/n_fit)*np.log(sma_fit/r0_fit))*n_fit

partial_list = np.array([pfpi,pfpr,pfpn])

Fisher_mat = np.ones((3,3))

for j in range(3):
    for k in range(3):
        Fisher_mat[j][k] = np.sum(partial_list[j]*partial_list[k]/lumi_err_fit**2)

cov_mat_total = np.linalg.inv(Fisher_mat)

# MCMC sample
I0_sample_mcmc = res.flatchain["I0"].values
r0_sample_mcmc = res.flatchain["r0"].values
n_sample_mcmc = res.flatchain["n"].values

sample_mcmc_list = [I0_sample_mcmc,r0_sample_mcmc,n_sample_mcmc]

'''
Inor_axis = np.linspace(-0.005,0.005,100)
rnor_axis = np.linspace(-0.005,0.005,100)
nnor_axis = np.linspace(-0.005,0.005,100)

Inor_grid,rnor_grid = np.meshgrid(Inor_axis,rnor_axis)
pdf_grid = eft.plot_contour_pdf(Inor_grid,rnor_grid,cov_mat)
'''

# 3 subplots
label_list = ["I0","r0","n"]
tvalue_list = [I0_fit, r0_fit, n_fit]
args_index_list = [(0,1),(0,2),(1,2)] 

fig = plt.figure(figsize=(12,12))
axes = [fig.add_subplot(221), fig.add_subplot(223), fig.add_subplot(224)]


for index_fig in range(3):
    ax = axes[index_fig]
    arg1_index, arg2_index = args_index_list[index_fig]
    cov_mat = np.array([[cov_mat_total[arg1_index][arg1_index],cov_mat_total[arg1_index][arg2_index]],
                        [cov_mat_total[arg2_index][arg1_index],cov_mat_total[arg2_index][arg2_index]]])

    # confidence ellipse
    x_1sig,y_1sig = eft.plot_confidence_ellipse(prob=0.6526,covmat=cov_mat)
    x_2sig,y_2sig = eft.plot_confidence_ellipse(prob=0.9544,covmat=cov_mat)
    x_3sig,y_3sig = eft.plot_confidence_ellipse(prob=0.9974,covmat=cov_mat)

    

    #MCMC samples
    arg1_sample_mcmc = sample_mcmc_list[arg1_index]
    arg2_sample_mcmc = sample_mcmc_list[arg2_index]

    arg1_fit = tvalue_list[arg1_index]
    arg2_fit = tvalue_list[arg2_index]

    arg1label = label_list[arg1_index]
    arg2label = label_list[arg2_index]

    # ax1 = ax.contourf((Inor_grid+1)*I0_fit,(rnor_grid+1)*r0_fit,pdf_grid)
    ax.plot((x_1sig+1)*arg1_fit,(y_1sig+1)*arg2_fit, c="red",    label="$1\sigma$ given by Fisher Matrix")
    ax.plot((x_2sig+1)*arg1_fit,(y_2sig+1)*arg2_fit, c="orange", label="$2\sigma$ given by Fisher Matrix")
    ax.plot((x_3sig+1)*arg1_fit,(y_3sig+1)*arg2_fit, c="green",  label="$3\sigma$ given by Fisher Matrix")
    ax.scatter(arg1_sample_mcmc,arg2_sample_mcmc,s=2,alpha=0.3,label="MCMC sampling",c="blue")
    ax.set_xlabel(arg1label)
    ax.set_ylabel(arg2label)

    # fig range
    x_center = np.mean((x_3sig+1)*arg1_fit)
    y_center = np.mean((y_3sig+1)*arg2_fit)
    x_radi = np.max((x_3sig+1)*arg1_fit)-np.min((x_3sig+1)*arg1_fit)
    y_radi = np.max((y_3sig+1)*arg2_fit)-np.min((y_3sig+1)*arg2_fit)

    ax.set_xlim(x_center-0.75*x_radi,x_center+0.75*x_radi)
    ax.set_ylim(y_center-0.75*y_radi,y_center+0.75*y_radi)

    if index_fig == 2:
        ax.legend()

ax_curve = fig.add_subplot(222)
ax_curve.plot(isolist.sma[num_drop_center_points:],modified_sersic(isolist.sma,I0_fit, r0_fit, n_fit)[num_drop_center_points:])
ax_curve.errorbar(isolist.sma, isolist.intens, yerr=isolist.int_err, fmt='o', markersize=2)
ax_curve.set_xlabel('sma')
ax_curve.set_ylabel('luminosity')
ax_curve.set_title("emcee, I0: %.4f, r0: %.2f, n: %.4f"%(I0_fit, r0_fit, n_fit, ))
plt.suptitle(GALAXY_NAME+" Modified sersic = $I_0\exp[-b(n)((r/r_0)^{1/n}-1)]$")
plt.savefig("test_pdf_fisher.pdf")
plt.close()

#KS test
'''
prob_nor = eft.check_sample_distribution(I0_sample_mcmc,r0_sample_mcmc,[I0_fit,r0_fit],cov_mat)

prob_list = np.linspace(0.01,0.99,100)
CDF_list = []
for prob in prob_list:
    CDF = np.sum(prob_nor<prob)/len(prob_nor)
    CDF_list.append(CDF)

fig,ax = plt.subplots(figsize=(6,6))
# ax1 = ax.contourf((Inor_grid+1)*I0_fit,(rnor_grid+1)*r0_fit,pdf_grid)
ax.plot(prob_list,CDF_list,label="CDF given by MCMC")
ax.plot(prob_list,prob_list,ls="--",label="ideal CDF given by Fisher matrix")
ax.set_xlabel('CDF(ideal)')
ax.set_ylabel('CDF(MCMC)')
# plt.colorbar(ax1)
ax.legend()
plt.savefig("test_fisher_CDF.pdf")
plt.close()
'''