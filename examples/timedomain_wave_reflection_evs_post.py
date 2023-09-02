import numpy as np
import matplotlib.pyplot as plt

if 1:
    resAlphaHrz = np.loadtxt("alpha_hrz.dat")
    resAlphaCon = np.loadtxt("alpha_con.dat")
    resEpsilonHrz = np.loadtxt("epsilon_hrz.dat")
    resEpsilonCon = np.loadtxt("epsilon_con.dat")

elif 0:
    resAlpha = np.loadtxt("alpha.dat")
    resEpsilon = np.loadtxt("epsilon.dat")
    resEpsilonLumped = np.loadtxt("epsilon_lumped.dat")
    resAlphaLumped = np.loadtxt("alpha_lumped.dat")
    #resEpsilonCorr = np.loadtxt("epsilon_corr.dat")
    #resEpsilonCorrMod = np.loadtxt("epsilon_corr_mod.dat")
else:
    resAlpha = np.loadtxt("alpha_1e-4.dat")
    resEpsilon = np.loadtxt("epsilon_1e-4.dat")
    resEpsilonCorr = np.loadtxt("epsilon_corr_1e-4.dat")
    resEpsilonCorrMod = np.loadtxt("epsilon_corr_mod_1e-4.dat")


plt.rcParams["figure.figsize"] = (10, 6)


plt.loglog(resAlphaCon[:, 1], resAlphaCon[:, 3], "-*", label="error alpha con", color="blue")
plt.loglog(resAlphaHrz[:, 1], resAlphaHrz[:, 3], "-*", label="error alpha hrz", color="orange")
plt.loglog(resEpsilonCon[:, 0], resEpsilonCon[:, 3], "-*", label="error epsilon con", color="red")
plt.loglog(resEpsilonHrz[:, 0], resEpsilonHrz[:, 3], "-*", label="error epsilon hrz", color="green")
#plt.loglog(resEpsilon[:, 0], resEpsilon[:, 3], "-*", label="error epsilon", color="orange")
#plt.loglog(resEpsilonCorr[:, 0], resEpsilonCorr[:, 3], "-*", label="error epsilon corrected", color="red")
#plt.loglog(resEpsilonCorrMod[:, 0], resEpsilonCorrMod[:, 3], "-*", label="error epsilon corrected mod", color="green")

plt.loglog(resAlphaCon[:, 1], resAlphaCon[:, -1], "--*", label="crit dt alpha con", color="blue")
plt.loglog(resAlphaHrz[:, 1], resAlphaHrz[:, -1], "--*", label="crit dt alpha hzr", color="orange")
plt.loglog(resEpsilonCon[:, 0], resEpsilonCon[:, -1], "--*", label="crit dt epsilon con", color="red")
plt.loglog(resEpsilonHrz[:, 0], resEpsilonHrz[:, -1], "--*", label="crit dt epsilon hzr", color="green")
#plt.loglog(resEpsilon[:, 0], resEpsilon[:, -1], "--*",  label="crit dt epsilon", color="orange")
#plt.loglog(resEpsilonCorr[:, 0], resEpsilonCorr[:, -1], "--*", label="crit dt epsilon corrected", color="red")
#plt.loglog(resEpsilonCorrMod[:, 0], resEpsilonCorrMod[:, -1], "--*", label="crit dt epsilon corrected mod", color="green")

plt.xlabel("epsilon / alpha")
plt.ylabel("error / crit dt")


plt.legend()
plt.title("Error dt=1.170300e-04")
plt.show()
