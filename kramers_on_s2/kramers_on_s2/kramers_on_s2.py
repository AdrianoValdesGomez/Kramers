"""Main module."""
import numpy as np


class KramersOnS2:
    """Class for simulating a particle on the sphere S2 with
    Kramers dynamics."""
    def __init__(self, potential, temperature,
                 friction, initial_condition,
                 time_step):
        self.potential = potential
        self.temperature = temperature
        self.friction = friction
        self.initial_condition = initial_condition
        self.time_step = time_step
        self.time = 0

    @staticmethod
    def orto(x):
        """Given a vector x, this function generates another one that
        is orthogonal with respect to the natural scalar product on 3
        dimensional euclidean space E3"""
        if np.dot(x, x) == 0:
            return 'No se puede: ese es el vector cero!'
        else:
            if 0 not in x:
                v1 = 1
                v2 = -(x[0]/x[1])
                v3 = 0
            else:
                if x[0] == 0:
                    if x[1] == 0:
                        v1 = 1
                        v2 = 0
                        v3 = 0
                    else:
                        v1 = 0
                        v2 = 0
                        v3 = 1
                elif x[1] == 0:
                    v1 = 0
                    v2 = 1
                    v3 = 0
                else:
                    v1 = 0
                    v2 = 0
                    v3 = 1
            return np.array([v1, v2, v3])

    @staticmethod
    def base_ort_nor(x):
        """This function generates a basis of
        vetors orthogonal, with respect to the
        natural scalar product on 3 dimensional
        euclidean space, to the vector x"""
        y = KramersOnS2.orto(x)
        v1 = y/np.linalg.norm(y)
        z = np.cross(x, v1)
        v2 = z/np.linalg.norm(z)
        return v1, v2

    @staticmethod
    def vector_des(v1, v2):
        """This function generates a unit random
        vetor, with unifom distribution among all
        possible directions,in the plane spanned
        by the linearly independent vectors v1
        and v2"""
        na = 2 * np.pi*np.random.rand()
        vn = v1*np.cos(na) + v2*np.sin(na)
        return vn/np.linalg.norm(vn)

    @staticmethod
    def vector_q(x, s):
        """This function returns a vector in the
        direction of x of length tan(s)"""
        q = np.tan(s)
        return q*x

    @staticmethod
    def nuevo_r(r, vector_q):
        """This function returns a point on the
        unit sphere, originaly in r, a geodesic
        distant s, in the direction
        of vector_q"""
        y = r + vector_q
        y = y/np.linalg.norm(y)
        return y

    @staticmethod
    def actualiza(r, s):
        """This function updates the position of
        a particleasembles one a function that
        generates a random vector in the plane
        orthogonal to r and of displacement
        magintude s"""
        v1, v2 = KramersOnS2.base_ort_nor(r)
        pre_q = KramersOnS2.vector_des(v1, v2)
        q = KramersOnS2.vector_q(pre_q, s)
        return KramersOnS2.nuevo_r(r, q)

    @staticmethod
    def var(D, delta_t):
        """This function returns the varianza of a
        brownian particle in the 2 dimensional
        infinite plane, of self difussion coeffient
        D, and in an interval of time of length
        delta_t"""
        return 4 * D * delta_t

    @staticmethod
    def ese(D, delta_t):
        """This function returns the magnitude of the
        random displacement displacement of a brownian
        particle that has a self-diffusion coeffient D,
        in a time interval of length delta_t"""
        return abs(np.random.normal(
            loc=0., scale=np.sqrt(
                KramersOnS2.var(D, delta_t)
                ), size=None))

    @staticmethod
    def act_n(lista, D, delta_t):
        """This function updates all the elements
        in a list using the actualiza function on
        each element"""
        l_particles = []
        for v in lista:
            s = KramersOnS2.ese(D, delta_t)
            l_particles.append(KramersOnS2.actualiza(v, s))
        return l_particles
