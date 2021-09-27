import taichi as ti
import numpy as np

from level_set import LevelSet
from differential import Differential

ti.init(arch=ti.gpu)

PI = 3.1415926535

# def diff_grid variable
dim = 2
diff_n_grid = 10
max_x = 1.0
min_x = .0
gradient = ti.Vector.field(dim, ti.f32, shape=(diff_n_grid, diff_n_grid))
divergence = ti.field(ti.f32, shape=(diff_n_grid, diff_n_grid))
laplacian = ti.field(ti.f32, shape=(diff_n_grid, diff_n_grid))
# v = ti.field(ti.f32, shape=(diff_n_grid, diff_n_grid))
grid_pos = ti.Vector.field(2, dtype=ti.f32, shape=(diff_n_grid, diff_n_grid))

# level set paramer
sign_distance_field = ti.field(ti.f32, shape=(diff_n_grid, diff_n_grid))

# def MPM variable
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 20000 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol = (dx * 0.5) ** 2
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
E_t, nu_t = 207, 0.3  # Young's modulus and Poisson's ratio
mu_t, lambda_t = E_t / (2 * (1 + nu_t)), E_t * nu_t / ((1 + nu_t) * (1 - 2 * nu_t))  # Lame parameters

# level set test
pos = ti.Vector.field(dim, ti.f32, shape=10)
value = ti.field(ti.f32, shape=10)

create_particle_num = ti.field(dtype=int, shape=())

particles = ti.Struct.field({
    "pos": ti.types.vector(2, ti.f32),
    "vel": ti.types.vector(2, ti.f32),
    "C": ti.types.matrix(2, 2, ti.f32),
    "F": ti.types.matrix(2, 2, ti.f32),
    "material": ti.i32,
    "mass": ti.f32
}, shape=n_particles)
nodes = ti.Struct.field({
    "node_v": ti.types.vector(2, ti.f32),
    "node_m": ti.f32
}, shape=(n_grid, n_grid))


# 将x复制给y
@ti.kernel
def copy_array(x: ti.template(), y: ti.template()):
    for I in ti.grouped(x):
        y[I] = x[I]


@ti.func
def get_random_pos(center_x, center_y, radius):
    u = ti.sqrt(ti.random()) * radius
    v = ti.random() * 2 * PI
    x = center_x + u * ti.cos(v)
    y = center_y + u * ti.sin(v)
    return [x, y]


@ti.kernel
def reset_nodes():
    for i, j in nodes:
        nodes[i, j].node_v = [0, 0]
        nodes[i, j].node_m = 0


@ti.kernel
def P2G():
    for p in range(create_particle_num[None]):  # Particle state update and scatter to grid (P2G)
        # 减0.5的作用是为了找到最近的网格节点
        base = (particles[p].pos * inv_dx - 0.5).cast(int)
        fx = particles[p].pos * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        particles[p].F = (ti.Matrix.identity(float, 2) + dt * particles[p].C) @ particles[
            p].F  # deformation gradient update
        h = 5.0  # Hardening coefficient: snow gets harder when compressed
        if particles[p].material == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if particles[p].material == 0:  # liquid
            mu = 0.0
        if particles[p].material == 2:
            la = lambda_t
            mu = mu_t
        U, sig, V = ti.svd(particles[p].F)
        J = 1.0
        for d in ti.static(range(2)):
            J *= sig[d, d]
        if particles[p].material == 0:  # Reset deformation gradient to avoid numerical instability
            particles[p].F = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        stress = 2 * mu * (particles[p].F - U @ V.transpose()) @ particles[p].F.transpose() + \
                 ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + particles[p].mass * particles[p].C
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            nodes[base + offset].node_v += weight * (particles[p].mass * particles[p].vel + affine @ dpos)
            nodes[base + offset].node_m += weight * particles[p].mass


@ti.kernel
def grid_operator():
    for i, j in nodes:
        if nodes[i, j].node_m > 0:  # No need for epsilon here
            nodes[i, j].node_v = (1 / nodes[i, j].node_m) * nodes[i, j].node_v  # Momentum to velocity
            nodes[i, j].node_v[1] -= dt * 50  # gravity
            if i < 3 and nodes[i, j].node_v[0] < 0:          nodes[i, j].node_v[0] = 0  # Boundary conditions
            if i > n_grid - 3 and nodes[i, j].node_v[0] > 0: nodes[i, j].node_v[0] = 0
            if j < 3 and nodes[i, j].node_v[1] < 0:          nodes[i, j].node_v[1] = 0
            if j > n_grid - 3 and nodes[i, j].node_v[1] > 0: nodes[i, j].node_v[1] = 0


@ti.kernel
def G2P():
    for p in range(create_particle_num[None]):  # grid to particle (G2P)
        base = (particles[p].pos * inv_dx - 0.5).cast(int)
        fx = particles[p].pos * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = nodes[base + ti.Vector([i, j])].node_v
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        particles[p].vel, particles[p].C = new_v, new_C
        if particles[p].material == 2 and particles[p].pos[0] < 0.1:
            particles[p].vel[0] = 0
        if particles[p].material == 2 and particles[p].pos[0] > 0.9:
            particles[p].vel[0] = 0
        particles[p].pos += dt * particles[p].vel  # advection


@ti.kernel
def initialize():
    for i in range(n_particles):
        particles[i].pos = [ti.random() * 0.3 + 0.2, ti.random() * 0.3 + 0.2]
        particles[i].material = 0
        particles[i].vel = ti.Vector([0, 0])
        particles[i].F = ti.Matrix([[1, 0], [0, 1]])
        particles[i].mass = p_vol * 1.0
    create_particle_num[None] += n_particles

    for i, j in ti.grouped(grid_pos):
        grid_pos[i, j] = ti.Vector([i / diff_n_grid, j / diff_n_grid])
    # for i in range(20000):
    #     # water
    #     particles[i].pos = [ti.random() * 0.984 + 0.007, ti.random() * 0.13 + 0.007]
    #     particles[i].material = 0
    #     particles[i].vel = ti.Vector([0, 0])
    #     particles[i].F = ti.Matrix([[1, 0], [0, 1]])
    #     particles[i].mass = p_vol * 1.0
    # create_particle_num[None] += 20000
    # for i in range(500):
    #     n = create_particle_num[None] + i
    #     particles[n].pos = get_random_pos(0.5, 0.22, 0.04)
    #     # x[n] = [ti.random() * 0.1 + 0.34+ 0.10, ti.random() * 0.008 + 0.31]
    #     particles[n].material = 1
    #     particles[n].vel = ti.Vector([0, 0])
    #     particles[n].F = ti.Matrix([[1, 0], [0, 1]])
    #     particles[n].mass = p_vol * 2.0
    # create_particle_num[None] += 500
    # for i in range(2000):
    #     n = create_particle_num[None] + i
    #     # mass_dx = 0.967 / 700
    #     particles[n].material = 2
    #     # x[n] = [i * mass_dx, 0.21]
    #     particles[n].pos = [ti.random() * 0.984 + 0.007, ti.random() * 0.0012 + 0.137]
    #     particles[n].vel = ti.Vector([0, 0])
    #     particles[n].F = ti.Matrix([[1, 0], [0, 1]])
    #     particles[n].mass = p_vol * 1.0
    # create_particle_num[None] += 2000


@ti.kernel
def init():
    for I in ti.grouped(pos):
        pos[I] = ti.Vector([ti.random(), ti.random()])


def main():
    initialize()
    init()
    value.fill(.0)

    level_set = LevelSet(diff_n_grid=diff_n_grid,
                         particle_pos=particles.pos,
                         sign_distance_field=sign_distance_field)
    # 生成符号距离场
    level_set.gen_level_set()
    level_set.bilinear_difference(pos, value)


    gui = ti.GUI('Surface Tension', res=512, background_color=0x112F41)

    # differential_solver = Differential(
    #     n_grid=diff_n_grid,
    #     min_x=min_x,
    #     max_x=max_x,
    #     field=v,
    #     dim=dim,
    #     gradient=gradient,
    #     divergence=divergence,
    #     laplacian=laplacian)
    # init()
    # differential_solver.central_difference()
    # copy_array(differential_solver.gradient, gradient)
    # differential_solver.central_divergence()
    # copy_array(differential_solver.divergence, divergence)
    # differential_solver.central_laplacian()
    # copy_array(differential_solver.laplacian, laplacian

    while gui.running:
        # for s in range(int(2e-3 // dt)):
        #     reset_nodes()
        #     P2G()
        #     grid_operator()
        #     G2P()
        colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
        # gui.circles(particles.pos.to_numpy(), radius=1.5, color=colors[particles.material.to_numpy()])
        gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk

        for i in range(diff_n_grid):
            gui.circles(grid_pos.to_numpy()[i], color=0xff66cc, radius=1.5)
            for j in range(diff_n_grid):
                gui.text(str(sign_distance_field.to_numpy()[i][j])[0: 5], grid_pos.to_numpy()[i][j], 10, color=0xffffff)

        gui.circles(pos.to_numpy(), color=0xffffff, radius=1.8)
        for i in range(10):
            gui.text(str(value[i])[0: 7], pos.to_numpy()[i], 12, 0xffffff)
    #             # x = str(round(gradient.to_numpy()[i][j][0], 3))
    #             # y = str(round(gradient.to_numpy()[i][j][1], 3))
    #             # s = x + "+" + y
    #             # gui.text(s,grid_pos.to_numpy()[i][j], 8, color=0xffffff)
    #     gui.show()


if __name__ == '__main__':
    main()
