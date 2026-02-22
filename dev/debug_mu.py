"""Debug script to reproduce the expected mu value for Beattie Whalley VLL test."""
import sys
sys.path.insert(0, '/home/caleb/Documents/University/CHE3123/thermo')
sys.path.insert(0, '/home/caleb/Documents/University/CHE3123/chemicals')

from thermo import *
from thermo.bulk import *
from fluids.numerics import assert_close
from fluids.two_phase_voidage import gas_liquid_viscosity
from chemicals.utils import Vm_to_rho, normalize
from math import exp, log

T, P = 298.15, 1e5
zs = [.25, 0.7, .05]

constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0],
                                     omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844],
                                     CASs=['71-36-3', '7732-18-5', '64-17-5'])

HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),]

ViscosityGases=[ViscosityGas(poly_fit=(390.65, 558.9, [4.166385860107714e-29, -1.859399624586853e-25, 3.723945144634823e-22, -4.410000193606962e-19, 3.412270901850386e-16, -1.7666632565075753e-13, 5.266250837132718e-11, 1.8202807683935545e-08, -3.7907568022643496e-07])),
ViscosityGas(poly_fit=(273.16, 1073.15, [-1.1818252575481647e-27, 6.659356591849417e-24, -1.5958127917299133e-20, 2.1139343137119052e-17, -1.6813187290802144e-14, 8.127448028541097e-12, -2.283481528583874e-09, 3.674008403495927e-07, -1.9313694390100466e-05])),
ViscosityGas(poly_fit=(300.0, 513.9, [2.7916394465461813e-24, -9.092375280175391e-21, 1.2862968526545343e-17, -1.032039387901207e-14, 5.13487008660069e-12, -1.6219017947521426e-09, 3.1752760767848214e-07, -3.51903254465602e-05, 0.0016941391616918362])),]

ViscosityLiquids=[ViscosityLiquid(exp_poly_fit=(190.0, 391.9, [1.8379049563136273e-17, -4.5666126233131545e-14, 4.9414486397781785e-11, -3.042378423089263e-08, 1.166244931040138e-05, -0.0028523723735774113, 0.4352378275340892, -37.99358630363772, 1456.8338572042996])),
ViscosityLiquid(exp_poly_fit=(273.17, 647.086, [-3.2967840446295976e-19, 1.083422738340624e-15, -1.5170905583877102e-12, 1.1751285808764222e-09, -5.453683174592268e-07, 0.00015251508129341616, -0.024118558027652552, 1.7440690494170135, -24.96090630337129])),
ViscosityLiquid(exp_poly_fit=(159.11, 514.7, [-2.0978513357499417e-18, 4.812669873819701e-15, -4.572016638774548e-12, 2.299873746519043e-09, -6.408737804647756e-07, 8.908272738941156e-05, -0.002254199305798619, -0.8783232122373867, 74.74147552003194])),]

ViscosityGasMixtureObj = ViscosityGasMixture(ViscosityGases=ViscosityGases, correct_pressure_pure=False, method=LINEAR)
ViscosityLiquidMixtureObj = ViscosityLiquidMixture(ViscosityLiquids=ViscosityLiquids, correct_pressure_pure=False, method=LINEAR)

correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, HeatCapacityGases=HeatCapacityGases,
                                           ViscosityLiquids=ViscosityLiquids, ViscosityGases=ViscosityGases,
                                           ViscosityGasMixtureObj=ViscosityGasMixtureObj,
                                           ViscosityLiquidMixtureObj=ViscosityLiquidMixtureObj)

eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)

T_VLL = 361.0
VLL_betas = [0.027939322463013245, 0.6139152961492603, 0.35814538138772645]
VLL_zs_gas = [0.23840099709086618, 0.5786839935180893, 0.18291500939104433]
VLL_zs_l0 = [7.619975052238078e-05, 0.9989622883894993, 0.0009615118599781685]
VLL_zs_l1 = [0.6793120076703765, 0.19699746328631032, 0.12369052904331329]
gas_VLL = gas.to(T=T_VLL, P=P, zs=VLL_zs_gas)
l0_VLL = liq.to(T=T_VLL, P=P, zs=VLL_zs_l0)
l1_VLL = liq.to(T=T_VLL, P=P, zs=VLL_zs_l1)

VLL_kwargs = dict(T=T_VLL, P=P, zs=zs,
                 gas=gas_VLL, liquids=[l0_VLL, l1_VLL], solids=[], betas=VLL_betas,
                 flash_specs=None, flash_convergence=None,
                 constants=constants, correlations=correlations, flasher=None)

settings = BulkSettings(mu_VL='Beattie Whalley', mu_LL=LOG_PROP_MASS_WEIGHTED)
obj = EquilibriumState(settings=settings, **VLL_kwargs)

mu_expect = 9.444973170122925e-05

# === Old path: what liquid_bulk produces ===
lb = obj.liquid_bulk
print("=== Old path (liquid_bulk) ===")
print(f"liquid_bulk.phase_fractions = {lb.phase_fractions}")
print(f"liquid_bulk.betas_mass = {lb.betas_mass}")
print(f"liquid_bulk.mu() = {lb.mu()}")
print(f"liquid_bulk.rho_mass() = {lb.rho_mass()}")

print("\n=== Phase properties ===")
print(f"gas.mu() = {gas_VLL.mu()}")
print(f"l0.mu() = {l0_VLL.mu()}")
print(f"l1.mu() = {l1_VLL.mu()}")
print(f"gas.rho_mass() = {gas_VLL.rho_mass()}")
print(f"l0.rho_mass() = {l0_VLL.rho_mass()}")
print(f"l1.rho_mass() = {l1_VLL.rho_mass()}")

print("\n=== Overall betas ===")
print(f"obj.betas_mass = {obj.betas_mass}")
x_old = obj.betas_mass[0]
print(f"x (gas mass frac) = {x_old}")

print("\n=== Reproduce expected value ===")
mug = gas_VLL.mu()
mul_old = lb.mu()
rhog = gas_VLL.rho_mass()
rhol_old = lb.rho_mass()
mu_old = gas_liquid_viscosity(x_old, mul_old, mug, rhol_old, rhog, Method='Beattie Whalley')
print(f"mu_old = {mu_old}")
print(f"mu_expect = {mu_expect}")
print(f"match old = {abs(mu_old - mu_expect)/mu_expect < 1e-10}")

print("\n=== New path: from overall bulk ===")
liquids = [l0_VLL, l1_VLL]
liquid_mole_betas_raw = VLL_betas[1:]
liquid_mole_betas_norm = normalize(liquid_mole_betas_raw)
print(f"liquid_mole_betas_raw = {liquid_mole_betas_raw}")
print(f"liquid_mole_betas_norm = {liquid_mole_betas_norm}")

# Mass betas via phase_subset_betas logic
mass_amounts = [b*p.MW() for b, p in zip(liquid_mole_betas_norm, liquids)]
tot_mass = sum(mass_amounts)
liquid_mass_betas = [m/tot_mass for m in mass_amounts]
print(f"liquid_mass_betas (new) = {liquid_mass_betas}")
print(f"liquid_bulk.betas_mass  = {lb.betas_mass}")

# mul with mass betas (LOG_PROP_MASS_WEIGHTED)
mul_new = exp(liquid_mass_betas[0]*log(l0_VLL.mu()) + liquid_mass_betas[1]*log(l1_VLL.mu()))
print(f"\nmul_old (liquid_bulk) = {mul_old}")
print(f"mul_new              = {mul_new}")

# rhol with MOLE betas
V_liquid = liquid_mole_betas_norm[0]*l0_VLL.V() + liquid_mole_betas_norm[1]*l1_VLL.V()
MW_liquid = liquid_mole_betas_norm[0]*l0_VLL.MW() + liquid_mole_betas_norm[1]*l1_VLL.MW()
rhol_new = Vm_to_rho(V_liquid, MW_liquid)
print(f"rhol_old (liquid_bulk) = {rhol_old}")
print(f"rhol_new               = {rhol_new}")

mu_new = gas_liquid_viscosity(x_old, mul_new, mug, rhol_new, rhog, Method='Beattie Whalley')
print(f"\nmu_new    = {mu_new}")
print(f"mu_expect = {mu_expect}")
print(f"match new = {abs(mu_new - mu_expect)/mu_expect < 1e-10}")

print("\n=== Debug rhol ===")
print(f"lb.V() = {lb.V()}")
print(f"lb.MW() = {lb.MW()}")
print(f"lb.rho_mass() from V,MW = {lb.MW()/(lb.V()*1000)}")
print(f"l0.V() = {l0_VLL.V()}")
print(f"l1.V() = {l1_VLL.V()}")

# Raw betas (unnormalized, as liquid_bulk has them)
raw = liquid_mole_betas_raw
V_raw = raw[0]*l0_VLL.V() + raw[1]*l1_VLL.V()
MW_raw = raw[0]*l0_VLL.MW() + raw[1]*l1_VLL.MW()
print(f"\nWith raw (unnorm) betas:")
print(f"  V_raw = {V_raw}, MW_raw = {MW_raw}")
print(f"  rho = {MW_raw/(V_raw*1000)}")

# Normalized betas
nrm = liquid_mole_betas_norm
V_nrm = nrm[0]*l0_VLL.V() + nrm[1]*l1_VLL.V()
MW_nrm = nrm[0]*l0_VLL.MW() + nrm[1]*l1_VLL.MW()
print(f"\nWith normalized betas:")
print(f"  V_nrm = {V_nrm}, MW_nrm = {MW_nrm}")
print(f"  rho = {MW_nrm/(V_nrm*1000)}")
print(f"  Vm_to_rho = {Vm_to_rho(V_nrm, MW_nrm)}")

print("\n=== The right way: MW from liquid_zs, V from betas ===")
# liquid_zs = normalize(sum of beta_i * zs_i)
liquid_zs = normalize([raw[0]*l0_VLL.zs[j] + raw[1]*l1_VLL.zs[j] for j in range(3)])
MW_from_zs = sum(z*mw for z, mw in zip(liquid_zs, constants.MWs))
V_from_betas = raw[0]*l0_VLL.V() + raw[1]*l1_VLL.V()
print(f"liquid_zs = {liquid_zs}")
print(f"lb.zs     = {lb.zs}")
print(f"MW_from_zs = {MW_from_zs}")
print(f"lb.MW()    = {lb.MW()}")
print(f"V_from_betas = {V_from_betas}")
print(f"lb.V()       = {lb.V()}")
rhol_correct = MW_from_zs / (V_from_betas * 1000)
print(f"rhol_correct = {rhol_correct}")
print(f"lb.rho_mass  = {lb.rho_mass()}")

mu_correct = gas_liquid_viscosity(x_old, mul_new, mug, rhol_correct, rhog, Method='Beattie Whalley')
print(f"\nmu_correct = {mu_correct}")
print(f"mu_expect  = {mu_expect}")
print(f"match = {abs(mu_correct - mu_expect)/mu_expect < 1e-10}")

print("\n=== What the current code actually computes ===")
# liquid_betas in current code are mass-weighted (from phase_subset_betas with LOG_PROP_MASS_WEIGHTED)
V_with_mass_betas = liquid_mass_betas[0]*l0_VLL.V() + liquid_mass_betas[1]*l1_VLL.V()
MW_with_mass_betas = liquid_mass_betas[0]*l0_VLL.MW() + liquid_mass_betas[1]*l1_VLL.MW()
rhol_mass_betas = Vm_to_rho(V_with_mass_betas, MW_with_mass_betas)
print(f"rhol with mass betas = {rhol_mass_betas}")

V_with_mole_betas = nrm[0]*l0_VLL.V() + nrm[1]*l1_VLL.V()
MW_with_mole_betas = nrm[0]*l0_VLL.MW() + nrm[1]*l1_VLL.MW()
rhol_mole_betas = Vm_to_rho(V_with_mole_betas, MW_with_mole_betas)
print(f"rhol with mole betas = {rhol_mole_betas}")

mu_from_code = gas_liquid_viscosity(x_old, mul_new, mug, rhol_mole_betas, rhog, Method='Beattie Whalley')
print(f"mu with mole betas rhol = {mu_from_code}")
print(f"This should be the new test value: {repr(mu_from_code)}")
