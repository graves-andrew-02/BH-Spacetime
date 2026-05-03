import numpy as np
from numba import jit
from tqdm import tqdm
import pickle
filename = 'log_G_hm_endurance'


# @title FDM 4th order
@jit
def dxy_4th_A(y, x):
    h = x[1] - x[0]
    dyx = np.zeros_like(y)

    # centered differences
    dyx[2:-2] = (-y[4:] + 8*y[3:-1] - 8*y[1:-3] +y[:-4]) / (12.0*h)

    dyx[0]    = (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] - 3*y[4])/ (12.0*h)
    dyx[1]    = (-25*y[1] + 48*y[2] - 36*y[3] + 16*y[4] - 3*y[5])/ (12.0*h)
    dyx[-1]    = (25*y[-1] - 48*y[-2] + 36*y[-3] - 16*y[-4] + 3*y[-5])/ (12.0*h)
    dyx[-2]    = (25*y[-2] - 48*y[-3] + 36*y[-4] - 16*y[-5] + 3*y[-6])/ (12.0*h)
    return dyx
@jit
def dxy_4th_A_2nd(y, x):
    h = x[1] - x[0]
    inv_h2 = 1.0 / (h*h)
    d2yx = np.zeros_like(y)

    d2yx[2:-2] = (-y[:-4] + 16*y[1:-3] - 30*y[2:-2] + 16*y[3:-1] - y[4:]) * (inv_h2/12)

    d2yx[-1]    = (45*y[-1] - 154*y[-2] + 214*y[-3] - 156*y[-4] + 61*y[-5] - 10*y[-6]) * (inv_h2/12)
    d2yx[0]    = (45*y[0] - 154*y[1] + 214*y[2] - 156*y[3] + 61*y[4] - 10*y[5]) * (inv_h2/12)
    d2yx[-2]    = (45*y[-2] - 154*y[-3] + 214*y[-4] - 156*y[-5] + 61*y[-6] - 10*y[-7]) * (inv_h2/12)
    d2yx[1]    = (45*y[1] - 154*y[2] + 214*y[3] - 156*y[4] + 61*y[5] - 10*y[6]) * (inv_h2/12)
    return d2yx

# @title FDM-2nd order and RK4

def RK4_ADM(y, dty, dt, x):
  k1 = dty(y, x)
  k2 = dty(y + 0.5*dt*k1, x)
  k3 = dty(y + 0.5*dt*k2, x)
  k4 = dty(y + dt*k3, x)

  y = y + (dt/6)*(k1+2*(k2+k3)+k4)
  return y

@jit
def dxy_2nd_A(y, x):
    h = x[1] - x[0]
    dyx = np.zeros_like(y)

    # centered differences
    dyx[1:-1] = (y[2:] - y[:-2]) / (2.0*h)
    dyx[0]    = (-3*y[0] +4*y[1] - y[2])/ (2.0*h) #forward differences
    dyx[-1]    = (25*y[-1] - 48*y[-2] + 36*y[-3] - 16*y[-4] + 3*y[-5])/ (12.0*h)

    return dyx
@jit
def dxy_2nd_A_2nd(y, x):
    h = x[1] - x[0]
    inv_h2 = 1.0 / (h*h)
    d2yx = np.zeros_like(y)
    y_ghost_right = - y[-1]
    d2yx[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) * inv_h2
    d2yx[0]    = (2*y[0] - 5*y[1] + 4*y[2] - y[3]) * inv_h2 #forward differences
    d2yx[-1]    = (2*y[-1] - 5*y[-2] + 4*y[-3] - y[-4]) * inv_h2 #backward differences

    return d2yx

@jit
def dxy_2nd_D(y, x, y_left=0.0, y_right=0.0):
    h = x[1] - x[0]
    dyx = np.zeros_like(y)

    # ghost points from Dirichlet BCs
    y_ghost_left = 2 * y_left - y[0]
    y_ghost_right = 2 * y_right - y[-1]

    # centered difference using ghost points
    dyx[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    dyx[0]    = (y[1] - y_ghost_left) / (2 * h)
    dyx[-1]   = (y_ghost_right - y[-2]) / (2 * h)
    return dyx

@jit
def dxy_2nd_D_2nd(y, x, y_left=0.0, y_right=0.0): # second order and second differential with Dirichlet BCs
    h = x[1] - x[0]
    dyx = np.zeros_like(y)
    inv_h2 = 1.0 / (h*h)

    # ghost points from Dirichlet BCs due to staggered grid
    y_ghost_left = 2 * y_left - y[0]
    y_ghost_right = 2 * y_right - y[-1]

    # centered difference using ghost points
    dyx[1:-1] = (y[2:] + y[:-2] - 2*y[1:-1]) * inv_h2
    dyx[0]    = (y[1] + y_ghost_left -2*y[0] ) * inv_h2
    dyx[-1]   = (y[-2] + y_ghost_right -2*y[-1] ) * inv_h2
    return dyx
@jit
def dxy_2nd_N(y, x, y_left=0.0, y_right=0.0):
    h = x[1] - x[0]
    dyx = np.zeros_like(y)

    # ghost points from Neumann BCs: g_left = (y1 - y_-1)/(2h), g_right = (y_N - y_{N-2})/(2h)
    #y_ghost_left  = y[1]  - 2.0*h*y_left
    y_ghost_right = 2.0*h*y_right + y[-2]

    # centered differences
    dyx[1:-1] = (y[2:] - y[:-2]) / (2.0*h)
    dyx[0]    = (y[1] - y[0])  / (2.0*h)
    #dyx[-1]   = (y_ghost_right - y[-2]) / (2.0*h)
    dyx[-1] = y_right
    return dyx


@jit
def dxy_2nd_N_2nd(y, x, y_left = 0.0, y_right = 0.0):
    h = x[1] - x[0]
    inv_h2 = 1.0 / (h*h)
    d2yx = np.zeros_like(y)

    # ghost points from Neumann BCs: g_left = (y1 - y_-1)/(2h), g_right = (y_N - y_{N-2})/(2h)
    y_ghost_left  = y[1]  - 2.0*h*y_left
    y_ghost_right = 2.0*h*y_right + y[-2]

    d2yx[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) * inv_h2
    d2yx[0]    = (y[1] -2*y[0] + y_ghost_left)  * inv_h2
    d2yx[-1]   = (y_ghost_right -2*y[-1] + y[-2]) * inv_h2
    return d2yx

#@title Staggered Grid and Helper functions
def make_staggeredx(xnum, end):
  x = np.linspace(0, (xnum-1)/xnum , xnum)
  x += 1/(2*xnum)
  x = x*end
  return x


def get_grad(vars, grid): # conformal version
  gradishs = np.ndarray(shape = [4,len(grid)])
  gradishs[0] = dxy_2nd_N(vars[0], grid)
  gradishs[1] = dxy_2nd_N(vars[1], grid)
  gradishs[2] = dxy_2nd_D(vars[2], grid, y_left= 0.0, y_right= 0.0)
  gradishs[3] = dxy_2nd_D(vars[3], grid, y_left= 0.0, y_right= 0.0)
  return (gradishs)

# @title Constraints
def get_constaints(vars, grid, mass = 1): # this version is for a conformal b
  rates = np.ndarray(shape = [6,len(grid)], dtype=np.double)
  lapse, shift = vars[4] , vars[5]
  gradishs = get_grad(vars, grid)

  r_squared = grid * grid


  ######Shit for ease later in function
  ####################################################################################################
  r_squared = grid * grid
  conformal_scaling = (1 + (mass/(2*grid)))
  chi = conformal_scaling**4
  chi_1st = -2*mass * (conformal_scaling**3) / r_squared
  chi_2nd = (mass/r_squared) * (((4*conformal_scaling**3)/grid) + ((3*mass*conformal_scaling**2)/r_squared))
  ####################################################################################################
  shift_1st = dxy_4th_A(shift, grid)
  gradishs = get_grad(vars, grid)
  lapse_1st = dxy_4th_A(vars[4], grid)
  lapse_2nd = dxy_4th_A_2nd(vars[4], grid)

  #rewrite a as a function of conformal stuff
  a = chi * vars[0]
  a_1st = chi_1st*vars[0] + chi*gradishs[0]

  #rewrite b as a function of conformal stuff
  b = chi * r_squared *vars[1]
  b_1st = (chi_1st * r_squared*vars[1]) + (chi* (2*grid*vars[1] + r_squared*gradishs[1])) # fine
  b_2nd = (chi_2nd*r_squared*vars[1]) + 2*(chi_1st*(2*grid*vars[1] + r_squared*gradishs[1])) + (chi*(2*vars[1] + 4*grid*gradishs[1] + r_squared*dxy_2nd_N_2nd(vars[1], grid)))

  #rewrite c as a function of conformal stuff
  c = chi * vars[2]
  c_1st = chi_1st*vars[2] + chi*gradishs[2]

  #rewrite d as a function of conformal stuff
  d = chi * r_squared *vars[3]
  d_1st = (chi_1st * r_squared*vars[3]) + (chi* (2*grid*vars[3] + r_squared*gradishs[3])) # fine


  #momentum constraint
  mom = ((c*b_1st /a)+(d*b_1st/b)-(2*d_1st)) / (b)
  ham = ((2*d**2)/(b**2)) + (((4*c*d)/(a)) + ((b_1st**2)/(2 *(a*b))) + ((a_1st*b_1st)/(a*a)) - (2*b_2nd/(a)) + (2))/b
  return mom, ham

#@title RATES THE IMPORTANT BIT
def get_rates(vars, grid, mass = 1): # this version is for a conformal a and b for isotropic schwartzchild
  lapse, shift = vars[4] , vars[5]
  rates = np.zeros_like(vars, dtype=np.double)

  ######Shit for ease later in function
  ####################################################################################################
  r_squared = grid * grid
  conformal_scaling = (1 + (mass/(2*grid)))
  chi = conformal_scaling**4
  chi_1st = -2*mass * (conformal_scaling**3) / r_squared
  chi_2nd = (mass/r_squared) * (((4*conformal_scaling**3)/grid) + ((3*mass*conformal_scaling**2)/r_squared))

  ####################################################################################################
  shift_1st = dxy_2nd_D(shift, grid)
  gradishs = get_grad(vars, grid)
  lapse_1st = dxy_2nd_N(vars[4], grid)#, y_left= 1.0, y_right= 1.0)
  lapse_2nd = dxy_2nd_N_2nd(vars[4], grid)#, y_left= 1.0, y_right= 1.0)

  #rewrite a as a function of conformal stuff
  a = chi * vars[0]
  a_1st = chi_1st*vars[0] + chi*gradishs[0]

  #rewrite b as a function of conformal stuff
  b = chi * r_squared *vars[1]
  b_1st = (chi_1st * r_squared*vars[1]) + (chi* (2*grid*vars[1] + r_squared*gradishs[1])) # fine
  b_2nd = (chi_2nd*r_squared*vars[1]) + 2*(chi_1st*(2*grid*vars[1] + r_squared*gradishs[1])) + (chi*(2*vars[1] + 4*grid*gradishs[1] + r_squared*dxy_2nd_N_2nd(vars[1], grid)))

  #rewrite c as a function of conformal stuff
  c = chi * vars[2]
  c_1st = chi_1st*vars[2] + chi*gradishs[2]

  #rewrite d as a function of conformal stuff
  d = chi * r_squared *vars[3]
  d_1st = (chi_1st * r_squared*vars[3]) + (chi* (2*grid*vars[3] + r_squared*gradishs[3])) # fine


  #actual PDEs
  rates[0] = (shift * a_1st) + 2*(a*shift_1st - lapse*c)
  rates[1] = (shift * b_1st) - (2*lapse*d)

  rates[2] = (
      lapse* (-(c*c/a) + ((2*d*c) + 0.5*((b_1st*b_1st)/b + (a_1st*b_1st)/a) - (b_2nd))/b)
      + (2*shift_1st*c) + (shift * c_1st) - lapse_2nd + 0.5*(a_1st*lapse_1st/a)
  )
  rates[3] = (1/a)*((c*d*lapse) + (0.25*a_1st*b_1st*lapse/a) - (0.5*b_2nd*lapse) - (0.5*lapse_1st*b_1st)) + lapse + shift*d_1st

  #change back to conformal rates
  inv_conf = 1/chi
  inv_confr2 = 1/(chi*r_squared)

  rates[0] *= inv_conf
  rates[1] *= inv_confr2
  rates[2] *= inv_conf
  rates[3] *= inv_confr2


  ### Shift/lapse and additional evolved variables ###

  K = (vars[2]/vars[0]) + 2*(vars[3]/(vars[1]))
  rates[4] = - (2*lapse *K)# + shift*lapse_1st # 2 or 1.46263
  b = vars[1]
  a = vars[0]
  c = vars[2]
  d = vars[3]

  #rates[4] = (lapse_2nd - ((c*c/(a*a)) + 2*(d*d/(b*b)))*lapse - 15*K)*0.2


  b_a = (vars[1]/vars[0])
  #rates[5] = -(2/grid)*(b_a**(2/3) - 1) - 2*((b_a**(-1/3))* (vars[0]*gradishs[1] - vars[1]*gradishs[0]))/(3*vars[0]*vars[0])# + shift*shift_1st
  #rates[5] *= 0.75
  rates[5] = - (b_a - 1)# + shift*shift_1st
  return (rates)


def initial_vars(xnum, end = 100):
  vars = np.ndarray(shape = [6,xnum], dtype=np.double) # a, b, c and d are the "ith" row

  x = make_staggeredx(xnum,end)

  vars[0] = np.ones(xnum, dtype=np.double)
  vars[1] = np.ones(xnum, dtype=np.double)
  vars[2] = np.zeros(xnum, dtype=np.double)
  vars[3] = np.zeros(xnum, dtype=np.double)
  vars[4] = np.full(xnum, 1.0, dtype=np.double)
  vars[5] = np.full(xnum, 0.0, dtype=np.double)#shift
  return vars, x

def save_simulation_data(filename, data_dict):
  """Saves a dictionary of simulation data to a pickle file."""
  with open(filename, 'wb') as f:
    pickle.dump(data_dict, f)
  print(f"Data saved successfully to {filename}")

def run(T_max, points, dt_dx_ratio = 0.1,saverate = 5, lengrid = 400):
  vars, x = initial_vars(points, end = lengrid)

  coord_relation = []
  a, b, c, d, lap, shit, lap_rate, shit_rate, momentum, hamiltonian, time, K = [], [], [], [], [], [], [], [], [], [], [], []
  dt = dt_dx_ratio*(x[1]-x[0])
  dt0 = dt_dx_ratio*(x[1]-x[0])
  t = 0
  next_save = 0.0#1/saverate

  print('Run starting')
  print('Resolution: {0:.4f}M - Cauchy Factor: {1:.4f}'.format(x[2]-x[1], dt_dx_ratio))
  pbar = tqdm(total=T_max + dt, desc="Progress")
  while (t < T_max):
    if not np.isnan(vars).any():
      vars = RK4_ADM(vars, get_rates, dt, x)
      rates = get_rates(vars,x)

      if any((vars[1]) <=0.0):
        crash = t
        break

      t += dt
    else:
      print(t)
      break

    pbar.update(dt)

    if t >= next_save:           # skip t = 0
        next_save += 1/saverate
        time.append(t)
        a.append(vars[0])
        b.append(vars[1])
        c.append(vars[2])
        d.append(vars[3])
        lap.append(vars[4])
        shit.append(vars[5])


        con = get_constaints(vars, x)
        momentum.append(con[0])
        hamiltonian.append(con[1])

  pbar.close()
  time.append(t)
  a.append(vars[0])
  b.append(vars[1])
  c.append(vars[2])
  d.append(vars[3])
  lap.append(vars[4])
  shit.append(vars[5])

  con = get_constaints(vars, x)
  momentum.append(con[0])
  hamiltonian.append(con[1])
  print('Run Complete')

  return np.array(x), a, b, c, d, lap, shit, lap_rate, shit_rate, momentum, hamiltonian, time, K


T = 301
sr = 4
ratio = .2
initial_resolution = 8000


x, a, b, c, d, lap, shit, lap_rate, shit_rate, momentum, hamiltonian, times, K=               run(T, 1* initial_resolution, dt_dx_ratio = ratio, saverate = sr)
x1, a1, b1, c1, d1, lap1, shit1, lap_rate1, shit_rate1, momentum1, hamiltonian1, times1, K1 = run(T, 2* initial_resolution, dt_dx_ratio = ratio, saverate = sr)
x2, a2, b2, c2, d2, lap2, shit2, lap_rate2, shit_rate2, momentum2, hamiltonian2, times2, K2 = run(T, 4* initial_resolution, dt_dx_ratio = ratio, saverate = sr)


simulation_data = {
    'x3': x , 'a3': a , 'b3': b , 'c3': c , 'd3': d , 'lap3': lap , 'shit3': shit , 'momentum3': momentum , 'hamiltonian3': hamiltonian , 'times3': times ,
    'x4': x1, 'a4': a1, 'b4': b1, 'c4': c1, 'd4': d1, 'lap4': lap1, 'shit4': shit1, 'momentum4': momentum1, 'hamiltonian4': hamiltonian1, 'times4': times1,
    'x5': x2, 'a5': a2, 'b5': b2, 'c5': c2, 'd5': d2, 'lap5': lap2, 'shit5': shit2, 'momentum5': momentum2, 'hamiltonian5': hamiltonian2, 'times5': times2
}


save_simulation_data(f'{filename}_1.pkl', simulation_data)