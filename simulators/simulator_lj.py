import numpy as np
import torch

class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def unit_vector(self):
        mag = self.magnitude()
        return self if mag == 0 else self / mag

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.force = Vector3D(0, 0, 0)
        self.mass = 1.0

    def update_position(self, dt):
        self.position = self.position + self.velocity * dt

    def update_velocity(self, dt_half):
        self.velocity = self.velocity + (self.force * (dt_half / self.mass))

def compute_forces(particles, epsilon, sigma):
    p1, p2 = particles
    r_vec = p1.position - p2.position
    r = r_vec.magnitude()
    if r == 0:
        return
    force_magnitude = 48 * epsilon / r * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6)
    force_ij = r_vec.unit_vector() * force_magnitude
    p1.force = force_ij
    p2.force = force_ij * -1

def velocity_verlet(particles, epsilon, sigma, dt, total_time):
    steps = int(total_time / dt)
    data = []
    for step in range(steps):
        for p in particles:
            p.update_velocity(dt / 2)
        for p in particles:
            p.update_position(dt)
        compute_forces(particles, epsilon, sigma)
        for p in particles:
            p.update_velocity(dt / 2)
        state = torch.tensor(
            [p.position.x for p in particles] + [p.position.y for p in particles] + [p.position.z for p in particles] +
            [p.velocity.x for p in particles] + [p.velocity.y for p in particles] + [p.velocity.z for p in particles], 
            dtype=torch.float32
        ).unsqueeze(0)
        data.append(state)
    return torch.cat(data)
