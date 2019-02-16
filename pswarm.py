import numpy as np
from collections import namedtuple
import warnings
import os
from tabulate import tabulate 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib 
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

Pstate = namedtuple('Pstate', 'iter position velocity value pos_best val_best')

class SwarmParam(namedtuple('SwarmParam', 'inertia cognitive social local')):
    @classmethod
    def make(cls, inertia=0.9, cognitive=2, social=2, local=2):
        return cls(inertia, cognitive, social, local)
    def __add__(self, other):
        return SwarmParam(self.inertia+other.inertia, 
                          self.cognitive+other.cognitive, 
                          self.social+other.social, 
                          self.local+other.local)
    def __radd__(self, other):
        return self + other 
    def __mul__(self, scaler):
        return SwarmParam(scaler*self.inertia, 
                          scaler*self.cognitive, 
                          scaler*self.social, 
                          scaler*self.local)
    def __rmul__(self, scaler):
        return self * scaler
    def __sub__(self, other):
        return SwarmParam(self.inertia-other.inertia, 
                          self.cognitive-other.cognitive, 
                          self.social-other.social, 
                          self.local-other.local)

class Particle:
    """An object to keep track of a single particle trajectory 
    through the optimization space"""

    def __init__(self, x0=None, bounds=None):
        if x0 is None:
            raise ValueError('A particle must have an initial position.')    
        self.position = np.array(x0)
        self.ndim = len(x0)
        self.velocity = np.random.uniform(low=-1, high=1, size=self.ndim)
        self.pos_best = x0
        self.val_best = np.inf 
        self.value = np.inf
        self.iter = 0
        self.bounds = bounds 
            
        self.trajectory = [Pstate(self.iter,
                                  self.position, 
                                  self.velocity,
                                  self.value,
                                  self.pos_best,
                                  self.val_best)]
        
    def get_fun_val(self, func, strategy='minimize'):
        '''Evaluate the error at the present position'''
        self.value = func(self.position)
        if strategy == 'maximize':
            self.value = -self.value

    def update_best_pos_val(self):
        if self.value < self.val_best:
            self.pos_best = self.position 
            self.val_best = self.value  

    def update_velocity(self, 
                        swarm_param=SwarmParam.make(),
                        global_pos_best=None, 
                        local_pos_best=None):
        r = np.random.random(size=3)
        
        if local_pos_best is not None:
            v_local = swarm_param.local * \
                      r[2] * (local_pos_best - self.position)
        else:
            v_local = 0
        
        if global_pos_best is not None:
            v_social = swarm_param.social * \
                       r[1] * (global_pos_best - self.position)
        else:
            v_social = 0
        
        v_cognitive = swarm_param.cognitive * \
                      r[0] * (self.pos_best - self.position)
        self.velocity = swarm_param.inertia * self.velocity \
                      + v_cognitive + v_social + v_local 
        
    def update_position(self):
        self.position = np.add(self.position, self.velocity, casting='unsafe')
        self.position = np.array([i if j[0] <= i <= j[1] else j[0] 
            if j[0] > i else j[1] for i, j in zip(self.position, self.bounds)])


    def update_traj(self):
        self.iter += 1
        self.trajectory += [Pstate(self.iter,
                                   self.position, 
                                   self.velocity,
                                   self.value,
                                   self.pos_best,
                                   self.val_best)]

    def get_dist(self, other):
        return np.linalg.norm(self.position - other.position)

    @property
    def part_state(self):
        return (self.position, self.velocity, 
                self.value, self.pos_best, self.val_best)

class ParticleSwarmOptimizer:
    '''Master class to oversee all the particles 
    and update them in each iteration step.'''
    def __init__(self,
                 scheme='fully_informed',
                 func=None,
                 obj='minimize',
                 N_particles=20,
                 max_iter=50,
                 num_neighbors=1,
                 search_space=None, 
                 swarm_param={'inertia': 0.9, 
                              'cognitive': 2,
                              'social': 2,
                              'local': 2},
                 decay={'inertia': 0, 
                        'cognitive': 0, 
                        'social': 0, 
                        'local': 0},
                 verbose=True,
                 fname=False):

        if scheme in ['fully_informed', 'globally_informed', 
                      'locally_informed', 'self_informed']:
            self.scheme = scheme 
        else:
            raise NotImplementedError('The requested scheme'
                                      ' is not implemented yet.')
        self.func = func 
        self.obj = obj
        self.search_space = search_space
        self.ndim = len(search_space)
        self.N_particles = N_particles 
        self.max_iter = max_iter
        self.iter = 0
        self.num_neighbors=num_neighbors
        if (self.num_neighbors > 1 and 
            self.scheme not in ['fully_informed', 'locally_informed']):
            warnings.warn('Your specified num_neighbors and scheme is '
                          'incompatible. Switching to fully informed scheme.')
            self.scheme = 'fully_informed'
        self.swarm_param = SwarmParam.make(**swarm_param)
        self.decay = SwarmParam.make(**decay)
        self.verbose = verbose
        if fname is not None:
            self.fname = fname
            if os.path.exists(fname):
                os.remove(fname)
        self.global_pos_best = np.inf 
        self.global_value_best = np.inf 
        if self.scheme in ['fully_informed', 'locally_informed']:
            self.local_pos_best = [np.inf for _ in range(self.N_particles)]
            self.local_value_best = [np.inf for _ in range(self.N_particles)] 


    def initialize_particles(self):
        #TODO need better initialization
        grid_points =  [np.linspace(*k, self.N_particles)
                            for k in self.search_space] 
        grid_coords = [[]]
        
        for gp in grid_points:
            grid_coords = [x + [y] for x in grid_coords for y in gp]
                               
        np.random.shuffle(grid_coords)
        grid_coords = grid_coords[:self.N_particles]
        self.swarm = [Particle(x0=i, 
                               bounds=self.search_space) 
                               for i in grid_coords]

    def run(self):
        self.initialize_particles() 
        while self.iter <= self.max_iter:
            if self.verbose:
                msg = 'iter: {0:4d}, '
                msg += 'best_value: {1:0.8f}, '
                msg += 'best_position: {2}'
                print(msg.format(self.iter, 
                                 self.global_value_best, 
                                 self.global_pos_best))
            for n, particles in enumerate(self.swarm):
                particles.get_fun_val(self.func, strategy=self.obj)
                particles.update_best_pos_val()

                if particles.value < self.global_value_best:
                    self.global_pos_best = particles.position
                    self.global_value_best = particles.value

                if self.scheme == 'fully_informed':
                    self.update_local_best()
                    particles.update_velocity(self.swarm_param,
                                              self.global_pos_best,
                                              self.local_pos_best[n])
                elif self.scheme == 'globally_informed':
                    particles.update_velocity(self.swarm_param,
                                              self.global_pos_best)
                elif self.scheme == 'locally_informed':
                    self.update_local_best()
                    particles.update_velocity(self.swarm_param,
                                              self.local_pos_best[n])
                elif self.scheme == 'self_informed':
                    particles.update_velocity(self.swarm_param) 
                
                particles.update_position()
                particles.update_traj()
                self.swarm_param_scheduler() 
            if self.fname:
                self.write_snapshot() 
            self.iter += 1

    def update_local_best(self):
        #TODO need better algorithm 
        local_pos_best = []
        local_value_best = []
        for i, leptons in enumerate(self.swarm):
            temp_data = [] 
            for j, bosons in enumerate(self.swarm):
                if i == j:
                    continue
                temp_data += [(leptons.get_dist(bosons), 
                               bosons.pos_best, 
                               bosons.val_best)]
            temp_data = sorted(temp_data, key=lambda x: x[0])
            temp_data = temp_data[:self.num_neighbors]
            temp_data = min(temp_data, key=lambda x: x[2])
            local_pos_best += [temp_data[1]]
            local_value_best += [temp_data[2]]
        
        for n in range(self.N_particles):
            if local_value_best[n] < self.local_value_best[n]: 
                self.local_pos_best[n] = local_pos_best[n]
                self.local_value_best[n] = local_value_best[n]  

    def swarm_param_scheduler(self):
        self.swarm_param -= self.iter * self.decay 
    
    def write_snapshot(self):
        with open(self.fname, 'a') as f:
            msg = 'Particle statistics after {} iteration:\n'
            f.write(msg.format(self.iter))
            plist = ['particle no.', 'position', 'velocity', 'value', 
                     'best_position', 'best_value']
            #f.write('{} {} {} {} {} {}\n'.format(*plist))
            data = []
            for i, particles in enumerate(self.swarm):
                #f.write('{} {} {} {} {} {}\n'.format(i, *particles.part_state))
                data += [[i, *particles.part_state]]
            f.write(tabulate(data, headers=plist, tablefmt='grid'))
            f.write('\nswarm param: {}\n'.format(self.swarm_param))
            f.write('Global best position: {}\n'.format(self.global_pos_best))
            f.write('Global best value: {}\n\n\n'.format(
                     self.global_value_best))
            
            if self.iter == self.max_iter:
                msg = '\nParticle statistics for the particle number {}:\n'
                for n in range(self.N_particles):
                    f.write(msg.format(n))
                    plist = ['iter no.', 'position', 'velocity', 'value', 
                             'best_position', 'best_value']
                    data = []
                    for i, traj in enumerate(self.swarm[n].trajectory):
                        data += [[i, traj.position, traj.velocity,
                                  traj.value, traj.pos_best, traj.val_best]]
                    f.write(tabulate(data, headers=plist, tablefmt='grid'))
                    f.write('\n')

    def animation(self, best_pos=[[0, 0, 0]]):
        if not self.ndim in [2, 3]:
            raise NotImplementedError('animation is supported only '
                                      'for dimension 2 and 3.')
        nfr = self.max_iter # Number of frames
        fps = 1 # Frame per sec

        xs = []
        ys = []
        zs = []

        for iter in range(self.max_iter):
            tx = []
            ty = []
            for particles in self.swarm:
                tx += [particles.trajectory[iter].position[0].tolist()]
                ty += [particles.trajectory[iter].position[1].tolist()]
            xs += [tx]
            ys += [ty]

        zs = [[0 for i in j] for j in xs]
         
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sct, = ax.plot([], [], [], "o", markersize=2)
        
        def update(ifrm, xa, ya, za):
            sct.set_data(xa[ifrm], ya[ifrm])
            sct.set_3d_properties(za[ifrm])
            plt.title('Frame: {}'.format(ifrm), y=1.08)
        
        ax.set_xlim(*self.search_space[0])
        ax.set_ylim(*self.search_space[1])
        ax.set_zlim(0, 5)
        for bp in best_pos:
            ax.scatter(*bp, s=10, color='green', marker='*')
            ax.text(*bp, s='Solution', fontsize=10)
        ani = animation.FuncAnimation(fig, update, nfr, 
                 fargs=(xs, ys, zs), interval=1000/fps)

        plt.show()
