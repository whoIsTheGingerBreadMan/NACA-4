import numpy as np
import matplotlib.pyplot as plt

def get_naca(naca_number,num_points):
    m,p,t = _split_number(naca_number)
    assert len(naca_number) == 4
    x_u, x_l = get_points(naca_number, num_points=num_points/2)
    y_tu = _get_thickness(t,x_u)
    y_tl = _get_thickness(t,x_l)
    y_cu = _get_camber(m,p,x_u)
    y_cl = _get_camber(m, p, x_l)
    y_u = y_cu+ y_tu
    y_l = y_cl - y_tl
    x_u = np.append(x_u,1)
    x_l = np.append(x_l, 1)
    y_cl = np.append(y_cl, 0)
    y_u = np.append(y_u, 0)
    y_l = np.append(y_l, 0)

    return x_u,y_u,x_l,y_l,x_l,y_cl

def _split_number(naca_number):
    naca_number = [int(d) for d in str(naca_number)]
    m =naca_number[0]/100  # the max camber location
    p = naca_number[1]/10
    t = int(str(naca_number[2])+str(naca_number[3]))/100
    return m,p,t

def _get_thickness(t,points):
    points = np.array(points)
    return 5 * t * (.2969 * np.sqrt(points) - .1260 * points - .3516 * points ** 2 + .2843 * points ** 3 - .1015 * points ** 4)

def _get_camber(m,p,points):
    y_c = np.zeros_like(points)
    if m==0 and p==0:
        return y_c
    else:
        for i,px in enumerate(points):
            if px<p:
                y_c[i] = m/(p**2)*(2*p*px-px**2)
            else:
                y_c[i] = (m/(1-p)**2)*((1-2*p) + 2*p*px-px**2)
        return y_c

def get_points(naca_number,num_points):
    m,p,t = _split_number(naca_number)
    eps = 1/num_points
    sample_points = np.linspace(0, 1-eps, num=num_points)+eps
    dt = (m / p ** 2) * (2 * p - 2*sample_points)
    dyc = 5*t*(.2969/(2*np.sqrt(sample_points))-.1260-.7032*sample_points+.8529*sample_points**2-.406*sample_points**3)

    dy_u = np.abs(dt + dyc)
    dy_l = np.abs(dt - dyc)
    sample_points = sample_points-eps
    p_u = np.cumsum(num_points*dy_u/sum(dy_u))
    p_l =  np.cumsum(num_points*dy_l/sum(dy_l))
    npc_u = 0 #num points collected upper
    npc_l = 0 #num points collected lower
    points_upper = []
    points_lower = []
    sample_points = np.append(sample_points,1)
    for i in range(1,int(num_points)):
        current_pos = sample_points[i-1]
        next_pos = sample_points[i]
        if np.ceil(p_u[i]+eps)-npc_u>0:
            npta = np.floor(p_u[i])-npc_u
            points_upper.extend(np.linspace(current_pos,next_pos,npta))
            npc_u = np.floor(p_u[i])
        if np.ceil(p_l[i]+eps)-npc_l>0:
            npta = np.floor(p_l[i])-npc_l
            points_lower.extend(np.linspace(current_pos,next_pos,npta))
            npc_l = np.floor(p_l[i])
    return points_upper,points_lower


def plot_naca_airfoil(naca_number,num_points=100,with_camber=False):
    x_u,y_u,x_l,y_l,x_c,y_c = get_naca(naca_number,num_points)
    plt.plot(x_u,y_u,'bo-')
    plt.plot(x_l, y_l,'ro-')
    plt.plot(x_c,y_c)

    plt.show()

if __name__ == '__main__':
    plot_naca_airfoil('2412',100)


