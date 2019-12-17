""" Raypy.py

Classes and function for ray tracing in heterogeneous medium with flow

modif

"""
from numpy import *
from scipy import *
from pylab import *
from scipy.integrate import odeint
import sympy as sym


def Sym2fct_c(C0):
    x, y, z = sym.symbols('x y z')
    dC0dx = sym.diff(C0, x)
    dC0dy = sym.diff(C0, y)
    dC0dz = sym.diff(C0, z)
    c0 = sym.lambdify([x, y, z], C0)
    dc0dx = sym.lambdify([x, y, z], dC0dx)
    dc0dy = sym.lambdify([x, y, z], dC0dy)
    dc0dz = sym.lambdify([x, y, z], dC0dz)
    return c0, dc0dx, dc0dy, dc0dz

def Sym2fct_U(Ux, Uy, Uz):
    x, y, z = sym.symbols('x y z')
    
    dUxdx = sym.diff(Ux, x)
    dUydx = sym.diff(Uy, x)
    dUzdx = sym.diff(Uz, x)
    
    dUxdy = sym.diff(Ux, y)
    dUydy = sym.diff(Uy, y)
    dUzdy = sym.diff(Uz, y)
    
    dUxdz = sym.diff(Ux, z)
    dUydz = sym.diff(Uy, z)
    dUzdz = sym.diff(Uz, z)
    ux = sym.lambdify([x, y, z], Ux)
    uy = sym.lambdify([x, y, z], Uy)
    uz = sym.lambdify([x, y, z], Uz)
    duxdx = sym.lambdify([x, y, z], dUxdx)
    duydx = sym.lambdify([x, y, z], dUydx)
    duzdx = sym.lambdify([x, y, z], dUzdx)
    
    duxdy = sym.lambdify([x, y, z], dUxdy)
    duydy = sym.lambdify([x, y, z], dUydy)
    duzdy = sym.lambdify([x, y, z], dUzdy)
    
    duxdz = sym.lambdify([x, y, z], dUxdz)
    duydz = sym.lambdify([x, y, z], dUydz)
    duzdz = sym.lambdify([x, y, z], dUzdz)
    return ux, uy, uz, duxdx, duydx, duzdx, duxdy, duydy, duzdy, duxdz, duydz, duzdz

def Nulle(x, y, z):
    f = 0
    return f


def TrajRayons(X, t, mil):  # defines the system of odes
    """
    Systeme differentiel
    """
    x = X[0]
    y = X[1]
    z = X[2]
    nx = X[3]
    ny = X[4]
    nz = X[5]
    A = [[1 - nx ** 2, -nx * ny, -nx * nz], [-ny * nx, 1 - ny ** 2, -ny * nz], [-nz * nx, -nz * ny, 1 - nz ** 2]]
    T1 = mil.duxdx(x, y, z) * nx + mil.duydx(x, y, z) * ny + mil.duzdx(x, y, z) * nz
    T2 = mil.duxdy(x, y, z) * nx + mil.duydy(x, y, z) * ny + mil.duzdy(x, y, z) * nz
    T3 = mil.duxdz(x, y, z) * nx + mil.duydz(x, y, z) * ny + mil.duzdz(x, y, z) * nz
    B = [-mil.dcdx(x, y, z) - T1, -mil.dcdy(x, y, z) - T2, -mil.dcdz(x, y, z) - T3]
    C = dot(A, B)
    f = [mil.ux(x, y, z) + mil.c(x, y, z) * nx, mil.uy(x, y, z) + mil.c(x, y, z) * ny,
         mil.uz(x, y, z) + mil.c(x, y, z) * nz, C[0], C[1], C[2]]
    return (f)


def InterpLin(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    return a, b


class Medium:
    """ class for medium properties
        ====> speed of sound
        ====> flow velocity
    """

    def __init__(self, c, dcdx=Nulle, dcdy=Nulle, dcdz=Nulle, NameSound='', ux=Nulle, uy=Nulle, uz=Nulle, duxdx=Nulle,
                 duydx=Nulle, duzdx=Nulle, duxdy=Nulle, duydy=Nulle, duzdy=Nulle, duxdz=Nulle, duydz=Nulle, duzdz=Nulle,
                 NameFlow=''):
        self.c = c
        self.dcdx = dcdx
        self.dcdy = dcdy
        self.dcdz = dcdz
        self.NameSound = NameSound
        self.NameFlow = NameFlow
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.duxdx = duxdx
        self.duydx = duydx
        self.duzdx = duzdx
        self.duxdy = duxdy
        self.duydy = duydy
        self.duzdy = duzdy
        self.duxdz = duxdz
        self.duydz = duydz
        self.duzdz = duzdz

    # tracés des profils de c0 et U
    def Plot(self, Zmin = -1000, Zmax=1000):
        N = 100
        Z = linspace(Zmin, Zmax, N)
        CCX = zeros((N,))
        CCY = zeros((N,))
        CCZ = zeros((N,))
        UUx = zeros((N,))
        UUy = zeros((N,))
        UUz = zeros((N,))
        DCC = zeros((N,))
        for i in range(N):
            CCX[i] = self.c(Z[i], 0, 0)
            CCY[i] = self.c(0, Z[i], 0)
            CCZ[i] = self.c(0, 0, Z[i])
            
            UUx[i] = 0
            UUz[i] = 0
            UUy[i] = self.ux(0, Z[i], 0)
            #UUx[i] = sqrt(self.ux(Z[i], 0, 0)**2+self.uy(Z[i], 0, 0)**2+self.uz(Z[i], 0, 0)**2)
            #UUy[i] = (self.ux(0, Z[i], 0)**2+self.uy(0, Z[i], 0)**2+self.uz(0, Z[i], 0)**2)
            #UUz[i] = sqrt(self.ux(0, 0, Z[i])**2+self.uy(0, 0, Z[i])**2+self.uz(0, 0, Z[i])**2)
            # DCX[i]=self.dcdx(0,0,Z[i])
        #			DCY[i]=self.dcdy(0,0,Z[i])
        #			DCZ[i]=self.dcdz(0,0,Z[i])
        figure(figsize=(12,4))
        subplot(131)
        plot(CCX, 0.001*Z, linewidth=2)
        xlabel(r'$c_0(x,0,0)$ [m/s]')
        ylabel(r'$X$ [km]')
        grid()
        subplot(132)
        plot(CCY, 0.001*Z, linewidth=2)
        xlabel(r'$c_0(0,y,0)$ [m/s]')
        ylabel(r'$Y$ [km]')
        grid()
        subplot(133)
        plot(CCZ, 0.001*Z, linewidth=2)
        xlabel(r'$c_0(0,0,z)$ [m/s]')
        ylabel(r'$Z$ [km]')
        grid()
        figure(figsize=(12,4))
        subplot(131)
        plot(UUx, 0.001*Z, linewidth=2)
        xlabel(r'$\bf{U}(x,0,0)$ [m/s]')
        ylabel(r'$X$ [km]')
        grid()
        subplot(132)
        plot(UUy, 0.001*Z, linewidth=2)
        xlabel(r'$\bf{U}(0,y,0)$ [m/s]')
        ylabel('Y [km]')
        grid()
        subplot(133)
        plot(UUx, 0.001*Z, linewidth=2)
        xlabel(r'$\bf{U}(0,0,z)$ [m/s]')
        ylabel(r'$Z$ [km]')
        grid()

class Point:
    """ class Point
        -----------
        contains coordinates and other stuff for one point
    """

    def __init__(self, x0=0, y0=0, z0=0):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

    def Plot(self, x0=0, y0=0, z0=0):
        plot([x0], [y0], 'r', linewidth=3)


class Normal:
    """ class Normal
        -----------
        contains coordinates and other stuff for one point
    """

    def __init__(self, Theta=0, Phi=0):
        self.nx0 = cos(Theta)
        self.ny0 = sin(Theta)
        self.nz0 = 0
        self.Norm = sqrt(self.nx0 ** 2 + self.ny0 ** 2 + self.nz0 ** 2)

    def ComputeNormalesSources(self, x0, y0, scale):
        """ en entree coordonnees d un point
            en sortie coordonnees d un point a pi/2
        """
        x1 = x0 + scale * self.nx0
        y1 = y0 + scale * self.ny0
        return x1, y1

    def Plot(self, x0=0, y0=0, z0=0, scale=1.0):
        x1 = self.nx0 * scale
        y1 = self.ny0 * scale
        plot([x0, x1], [y0, y1], 'k', linewidth=3)


class Ray:
    """ class for one ray
    """

    def __init__(self, PointX0, NormalX0, Tfin, Nt):
        """ Initialisation of class Ray
            Input :
            PointX0  => type class Point
            NormalX0 => 	type class Vec
        """
        self.PointX0 = PointX0
        self.NormalX0 = NormalX0
        self.Tfin = Tfin
        self.Nt = Nt
        self.t = linspace(0, Tfin, Nt)
        self.Traj = zeros_like(self.t)

    def Integrate(self, mil):
        X0 = [self.PointX0.x0, self.PointX0.y0, self.PointX0.z0, self.NormalX0.nx0, self.NormalX0.ny0,
              self.NormalX0.nz0]
        self.Traj = odeint(TrajRayons, X0, self.t, args=(mil,))

    def IntegrateWithAnObstacleForward(self, mil, PosFrontx, MediumIn):
        X0 = [self.PointX0.x0, self.PointX0.y0, self.PointX0.z0, self.NormalX0.nx0, self.NormalX0.ny0,
              self.NormalX0.nz0]
        self.Traj = odeint(TrajRayons, X0, self.t, args=(mil,))
        # pour chaque pas de temps on test si on est dans l'obstacle ou pas
        #for n in range(lent(self.t)):
        cond = 1
        count = 0
        while cond == 1:
            if MediumIn(self.Traj[count,0], self.Traj[count,1])==1:
                count += 1
            else:
                cond = 0
                pass
        # Alors on a une reflection au temps count ...


        count = self.Nt - 1
        print('reflexion ?')
        print(self.Traj[count, 0], self.Traj[count, 1], MediumIn(self.Traj[count, 0], self.Traj[count, 1]))
        if (MediumIn(self.Traj[count, 0], self.Traj[count, 1]) == -1):
            print('dans la mauvaise zone')
            while ((MediumIn(self.Traj[count, 0], self.Traj[count, 1]) == -1) and (count >= 0)):
                count = count - 1
            LPX = self.Traj[count, 0]
            LPY = self.Traj[count, 1]
            OPX = self.Traj[count + 1, 0]
            OPY = self.Traj[count + 1, 1]
            a, b = InterpLin(LPX, LPY, OPX, OPY)
            # calcul du nouveau point de depart
            px = PosFrontx
            py = a * PosFrontx + b
            # calcul des normales
            nnx = -self.Traj[count, 3]
            nny = self.Traj[count, 4]
            ttemp = zeros((self.Nt - count,))
            ttemp = self.t[count:self.Nt - 1]
            X0 = [px, py, 0.0, nnx, nny, self.NormalX0.nz0]
            Traj = odeint(TrajRayons, X0, ttemp, args=(mil,))
            self.Traj[count:self.Nt - 1, :] = Traj[:, :]

    def IntegrateWithAVerticalFront(self, mil, PosFrontx, MediumIn):

        X0 = [self.PointX0.x0, self.PointX0.y0, self.PointX0.z0, self.NormalX0.nx0, self.NormalX0.ny0,
              self.NormalX0.nz0]
        count = 1
        ref = 0
        list_traj=[]
        t0 = 0
#        print(50*"*-")
#        print("Raypy")
#        print(len(self.t))
        for n in range(1,len(self.t)):
            t = self.t[t0:n]    
            Traj = odeint(TrajRayons, X0, t, args=(mil,))
            if MediumIn(Traj[-1, 0], Traj[-1, 1])==-1:
                #print(Traj[-1, 0], Traj[-1, 1], Traj[-1, 2], Traj[-1, 3], Traj[-1, 4], Traj[-1, 5])
                #print(Traj[-1, 0], Traj[-1, 1], MediumIn(Traj[-1, 0], Traj[-1, 1]))    
                list_traj.append(Traj)
                X0 = [Traj[-2, 0], Traj[-2, 1], Traj[-2, 2], -Traj[-2, 3], Traj[-2, 4], Traj[-2, 5]]
                t0 = n
            else:
                count=count + 1
        list_traj.append(Traj)
        # reconstruction de la trajectoire
        self.Traj = list_traj[0]
        for traj in list_traj[1:]:
            #print("multi traj", np.shape(traj), count)
            self.Traj = np.concatenate([self.Traj, traj])

    def IntegrateWithAHorizontalFront(self, mil, PosFrontx, MediumIn):

        X0 = [self.PointX0.x0, self.PointX0.y0, self.PointX0.z0, self.NormalX0.nx0, self.NormalX0.ny0,
              self.NormalX0.nz0]
        count = 1
        ref = 0
        list_traj=[]
        t0 = 0
        for n in range(1,len(self.t)):
            t = self.t[t0:n]    
            Traj = odeint(TrajRayons, X0, t, args=(mil,))
            if MediumIn(Traj[-1, 0], Traj[-1, 1])==-1:
                list_traj.append(Traj)
                X0 = [Traj[-2, 0], Traj[-2, 1], Traj[-2, 2], Traj[-2, 3], -Traj[-2, 4], Traj[-2, 5]]
                X1 = [Traj[-1, 0], Traj[-1, 1], Traj[-1, 2], Traj[-1, 3], -Traj[-1, 4], Traj[-1, 5]]                
                t0 = n
                
            else:
                count=count + 1
        list_traj.append(Traj)
        # reconstruction de la trajectoire
        self.Traj = list_traj[0]
        for traj in list_traj[1:]:
            #print("multi traj", np.shape(traj), count)
            self.Traj = np.concatenate([self.Traj, traj])
    def IntegrateWithGround(self, mil, PosFrontx, MediumIn):

        X0 = [self.PointX0.x0, self.PointX0.y0, self.PointX0.z0, self.NormalX0.nx0, self.NormalX0.ny0,
              self.NormalX0.nz0]
        count = 0
        ref = 0
        list_traj=[]
        t0 = 0
        #for n in range(1,len(self.t)):
        #t = self.t[t0:n]    
        
        while t0<len(self.t) and count <10:
            t = self.t[t0:len(self.t)] 
            Traj = odeint(TrajRayons, X0, t, args=(mil,))
            I = np.where(np.isnan(Traj[:,1])==True)
            if np.size(I[0])==0:
                t0 = len(self.t)
                list_traj.append(Traj)
            else:
#                print("ground detected", count)
#                print(I[0][0])
                t0 = I[0][0]-1
                Traj_temp= Traj[:t0+1,:]
                list_traj.append(Traj_temp)
            
                X0 = [Traj[t0, 0], Traj[t0, 1], Traj[t0, 2], Traj[t0, 3], -Traj[t0, 4], Traj[t0, 5]]
                if Traj[t0, 4]>0:
                    count = 100
                else:
                    count = count +1
#                print(t0)
#                print(X0)

        # reconstruction de la trajectoire
        self.Traj = list_traj[0]
        for traj in list_traj[1:]:
            #print("multi traj", np.shape(traj), count)
            self.Traj = np.concatenate([self.Traj, traj])    

        # on va faire les modifs ici car plotR2 ne fonctionne pas
        #self.Traj = Traj
    def PlotR(self, NumFig):
        figure(NumFig)
        plot(self.Traj[:, 0], self.Traj[:, 1], linewidth=3)
        grid()
        


#    tester autres traces
#    """
#    def PlotR2(self, NumFig):
#        figure(NumFig)
#        plot(self.Traj[:, 0], self.Traj[:, 1], 'r', linewidth=3)
#        grid()
     
    
    
class Source:

    def __init__(self, X, N, name, Tfin, Nt):
        """
            X  liste des points de departs des rayons
            N  liste des normales a t=0 des rayons
        """
        self.X = X
        self.N = N
        self.NbRay = len(X)
        self.name = name
        self.Rays = [] # ?
        self.Tfin = Tfin
        self.Nt = Nt
        for i in range(self.NbRay):
            self.Rays.append(Ray(X[i], N[i], Tfin=Tfin, Nt=Nt))

    def PlotSource(self, NumFig):
        figure(NumFig)
        # for i in range(self.NbRay):
        #	self.N[i].Plot(self.X[0].x0,self.X[0].y0)
        xx = zeros((self.NbRay,))
        yy = zeros((self.NbRay,))
        for i in range(self.NbRay):
            #self.X[i].Plot(self.X[i].x0,self.X[i].y0)
            xx[i] = self.X[i].x0
            yy[i] = self.X[i].y0
        plot(xx, yy, 'r', lw=3)
        plot(xx, yy, 'or', lw=3)
        xlabel('x')
        ylabel('y')

    def PlotNormalesSource(self, NumFig, scale):
        xx = zeros((self.NbRay,))
        yy = zeros((self.NbRay,))
        for i in range(self.NbRay):
            xx[i], yy[i] = self.N[i].ComputeNormalesSources(self.X[i].x0, self.X[i].y0, scale)
            plot([self.X[i].x0, xx[i]], [self.X[i].y0, yy[i]], 'b')
            grid('on')
        #axis([-0.1, 0.1, -0.1, 0.1])

        # self.N[i].Plot(self.X[i].x0,self.X[i].y0,scale=scale)

    def PlotRays(self, NumFig):
        figure(NumFig)
        for i in range(self.NbRay):
            self.Rays[i].PlotR(NumFig)
        xlabel('x [m]')
        ylabel('y [m]')
        grid()
        #axis([-0.1, 0.2, -0.1, 0.2])
        
        # pour tracer plotR2
    def PlotRays2(self, NumFig):
        figure(NumFig)
        for i in range(self.NbRay):
            self.Rays[i].PlotR2(NumFig)
        xlabel('x')
        ylabel('y')
        grid()
        #axis([-0.1, 0.2, -0.1, 0.2])

    def Propagate(self, Mil):
        for i in range(self.NbRay):
            self.Rays[i].Integrate(Mil)

    def PropagateWithAVerticalFront(self, Mil, PosFrontx, MediumIn):
        for i in range(self.NbRay):
            self.Rays[i].IntegrateWithAVerticalFront(Mil, PosFrontx, MediumIn)

    def PropagateWithAHorizontalFront(self, Mil, PosFrontx, MediumIn):
        for i in range(self.NbRay):
            self.Rays[i].IntegrateWithAHorizontalFront(Mil, PosFrontx, MediumIn)
    
    def PropagateWithGround(self, Mil, PosFrontx, MediumIn):
        for i in range(self.NbRay):
            print("Calcul du rayon", i)
            self.Rays[i].IntegrateWithGround(Mil, PosFrontx, MediumIn)


    def SaveFile(self, name):
        nom = name + '.csv'
        print('sauvegarde des resultats dans le fichier ' + nom)
        file = open(nom, 'w')
        file.write('X,Y,Z,T\n')
        for i in range(self.NbRay):
            for k in range(self.Nt):
                chaine = str(self.Rays[i].Traj[k, 0]) + ',' + str(self.Rays[i].Traj[k, 1]) + ',' + str(
                    self.Rays[i].Traj[k, 2]) + ',' + str(self.Rays[i].t[k]) + '\n'
                file.write(chaine)
        file.close()


class Rectangle:
    """ class Rectangle Obstacle
        -----------
        contains coordinates and other stuff for one point
    """

    def __init__(self, x0=0, y0=0, z0=0, x1=0.0, y1=0, z1=0):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1

    def Plot(self, NumFig):
        figure(NumFig)
        plot([self.x0, self.x0], [self.y0, self.y1], 'k', linewidth=7)
        plot([self.x1, self.x1], [self.y0, self.y1], 'k', linewidth=7)
        plot([self.x0, self.x1], [self.y0, self.y0], 'k', linewidth=7)
        plot([self.x0, self.x1], [self.y1, self.y1], 'k', linewidth=7)

# class Rays:
#""" class for all the individual rays
#"""
#	def __init__(self):

