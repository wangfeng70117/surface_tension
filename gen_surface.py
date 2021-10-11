import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

PI = 3.1415926535

# def diff_grid variable
dim = 2
diff_n_grid = 100
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
n_particles, n_grid = 40000 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol = (dx * 0.5) ** 2
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
E_t, nu_t = 207, 0.3  # Young's modulus and Poisson's ratio
mu_t, lambda_t = E_t / (2 * (1 + nu_t)), E_t * nu_t / ((1 + nu_t) * (1 - 2 * nu_t))  # Lame parameters

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

_et = np.array(
    [
        [[-1, -1], [-1, -1]],  #
        [[0, 1], [-1, -1]],  # a
        [[0, 2], [-1, -1]],  # b
        [[1, 2], [-1, -1]],  # ab
        [[1, 3], [-1, -1]],  # c
        [[0, 3], [-1, -1]],  # ca
        [[1, 3], [0, 2]],  # cb
        [[2, 3], [-1, -1]],  # cab
        [[2, 3], [-1, -1]],  # d
        [[2, 3], [0, 1]],  # da
        [[0, 3], [-1, -1]],  # db
        [[1, 3], [-1, -1]],  # dab
        [[1, 2], [-1, -1]],  # dc
        [[0, 2], [-1, -1]],  # dca
        [[0, 1], [-1, -1]],  # dcb
        [[-1, -1], [-1, -1]],  # dcab
    ],
    np.int32)
et = ti.Vector.field(2, int, _et.shape[:2])
et.from_numpy(_et)
edge = ti.Struct.field({
    "begin_point": ti.types.vector(2, ti.f32),
    "end_point": ti.types.vector(2, ti.f32)
}, shape=diff_n_grid ** 2)
edge_num = ti.field(int, shape=())
diff_dx = 1 / diff_n_grid
diff_inv_dx = 1 / diff_dx
radius = diff_dx


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
            if i < 5 and nodes[i, j].node_v[0] < 0:          nodes[i, j].node_v[0] = 0  # Boundary conditions
            if i > n_grid - 5 and nodes[i, j].node_v[0] > 0: nodes[i, j].node_v[0] = 0
            if j < 5 and nodes[i, j].node_v[1] < 0:          nodes[i, j].node_v[1] = 0
            if j > n_grid - 5 and nodes[i, j].node_v[1] > 0: nodes[i, j].node_v[1] = 0


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


# 生成level set隐式曲面
@ti.kernel
def gen_level_set():
    for i, j in ti.ndrange(diff_n_grid, diff_n_grid):
        min_dis = 10.0
        node_pos = ti.Vector([i * diff_dx, j * diff_dx])
        for I in range(particles.pos.shape[0]):
            distance = (particles.pos[I] - node_pos).norm() - radius
            if distance < min_dis:
                min_dis = distance
        sign_distance_field[i, j] = min_dis


@ti.func
def ti_vector(x, y):
    return ti.Vector([x, y])


@ti.func
def gen_edge_pos(i, j, e):
    a = sign_distance_field[i, j]
    b = sign_distance_field[i + 1, j]
    c = sign_distance_field[i, j + 1]
    d = sign_distance_field[i + 1, j + 1]
    base_grid_pos = diff_dx * ti.Vector([i, j])
    result_pos = ti.Vector([.0, .0])
    if e == 0:
        result_pos = base_grid_pos + ti.Vector([(abs(a) / (abs(a) + abs(b))) * diff_dx, 0])
    if e == 1:
        result_pos = base_grid_pos + ti.Vector([0, (abs(a) / (abs(a) + abs(c))) * diff_dx])
    if e == 2:
        result_pos = base_grid_pos + ti.Vector([diff_dx, (abs(b) / (abs(b) + abs(d))) * diff_dx])
    if e == 3:
        result_pos = base_grid_pos + ti.Vector([(abs(c) / (abs(c) + abs(d))) * diff_dx, diff_dx])
    return result_pos


# 将隐式曲面通过marching cube转化为显示曲面
@ti.kernel
def implicit_to_explicit():
    for i, j in ti.ndrange(diff_n_grid - 1, diff_n_grid - 1):
        id = 0
        if sign_distance_field[i, j] > 0: id |= 1
        if sign_distance_field[i + 1, j] > 0: id |= 2
        if sign_distance_field[i, j + 1] > 0: id |= 4
        if sign_distance_field[i + 1, j + 1] > 0: id |= 8
        for k in ti.static(range(2)):
            if et[id, k][0] != -1:
                n = ti.atomic_add(edge_num[None], 1)
                edge[n].begin_point = gen_edge_pos(i, j, et[id, k][0])
                edge[n].end_point = gen_edge_pos(i, j, et[id, k][1])


def main():
    initialize()

    gui = ti.GUI('Surface Tension', res=512, background_color=0x112F41)

    while gui.running:
        for s in range(int(2e-3 // dt) * 2):
            reset_nodes()
            P2G()
            grid_operator()
            G2P()
        gen_level_set()
        implicit_to_explicit()
        begin_point = edge.begin_point.to_numpy()[:edge_num[None]]
        end_point = edge.end_point.to_numpy()[:edge_num[None]]

        colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
        # gui.circles(particles.pos.to_numpy(), radius=1.5, color=colors[particles.material.to_numpy()])

        gui.lines(begin_point, end_point, color=0xff66cc, radius=1.5)
        edge_num[None] = 0
        # gui.show(f'{gui.frame:06d}.png')
        gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk


if __name__ == '__main__':
    main()
