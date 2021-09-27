import taichi as ti


@ti.data_oriented
class LevelSet:
    def __init__(self,
                 diff_n_grid=128,
                 particle_pos=None,
                 sign_distance_field=None
                 ):
        self.particle_pos = particle_pos
        self.sign_distance_field = sign_distance_field
        self.diff_n_grid = diff_n_grid
        self.dx = 1 / diff_n_grid
        self.inv_dx = 1 / self.dx
        self.radius = 0.5 * self.dx

    @ti.kernel
    def gen_level_set(self):
        for i, j in ti.ndrange(self.diff_n_grid, self.diff_n_grid):
            min_dis = 10.0
            node_pos = ti.Vector([i * self.dx, j * self.dx])
            for I in range(20000):
                distance = (self.particle_pos[I] - node_pos).norm() - self.radius
                if distance < min_dis:
                    min_dis = distance
            self.sign_distance_field[i, j] = min_dis

    # 双线性差值函数
    @ti.kernel
    def bilinear_difference(self, pos: ti.template(), value: ti.template()):
        for I in ti.grouped(pos):
            base = (pos[I] * self.inv_dx).cast(int)
            fx = pos[I] * self.inv_dx - base.cast(float)
            print(fx)
            w = [(1 - fx) * self.dx, fx * self.dx]
            new_value = .0
            for i, j in ti.static(ti.ndrange(2, 2)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1] * self.inv_dx * self.inv_dx
                new_value += self.sign_distance_field[base + offset] * weight
            value[I] = new_value
