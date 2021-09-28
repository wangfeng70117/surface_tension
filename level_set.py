import taichi as ti
import numpy as np


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
        self.radius = 0.4 * self.dx
        self._et = np.array(
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
        self.et = ti.Vector.field(2, int, self._et.shape[:2])
        self.et.from_numpy(self._et)
        self.edge = ti.Struct.field({
            "begin_point": ti.types.vector(2, ti.f32),
            "end_point": ti.types.vector(2, ti.f32)
        }, shape=diff_n_grid ** 2)
        self.edge_num = ti.field(int, shape=())

    # 生成level set隐式曲面
    @ti.kernel
    def gen_level_set(self):
        for i, j in ti.ndrange(self.diff_n_grid, self.diff_n_grid):
            min_dis = 10.0
            node_pos = ti.Vector([i * self.dx, j * self.dx])
            for I in range(self.particle_pos.shape[0]):
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
            w = [(1 - fx) * self.dx, fx * self.dx]
            new_value = .0
            for i, j in ti.static(ti.ndrange(2, 2)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1] * self.inv_dx * self.inv_dx
                new_value += self.sign_distance_field[base + offset] * weight
            value[I] = new_value

    @ti.func
    def ti_vector(self, x, y):
        return ti.Vector([x, y])

    @ti.func
    def gen_edge_pos(self, i, j, e):
        a = self.sign_distance_field[i, j]
        b = self.sign_distance_field[i + 1, j]
        c = self.sign_distance_field[i, j + 1]
        d = self.sign_distance_field[i + 1, j + 1]
        base_grid_pos = self.dx * ti.Vector([i, j])
        result_pos = ti.Vector([.0, .0])
        if e == 0:
            result_pos = base_grid_pos + ti.Vector([(abs(a) / (abs(a) + abs(b))) * self.dx, 0])
        if e == 1:
            result_pos = base_grid_pos + ti.Vector([0, (abs(a) / (abs(a) + abs(c))) * self.dx])
        if e == 2:
            result_pos = base_grid_pos + ti.Vector([self.dx, (abs(b) / (abs(b) + abs(d))) * self.dx])
        if e == 3:
            result_pos = base_grid_pos + ti.Vector([(abs(c) / (abs(c) + abs(d))) * self.dx, self.dx])
        return result_pos

    # 将隐式曲面通过marching cube转化为显示曲面
    @ti.kernel
    def implicit_to_explicit(self):
        for i, j in ti.ndrange(self.diff_n_grid - 1, self.diff_n_grid - 1):
            id = 0
            if self.sign_distance_field[i, j] > 0: id |= 1
            if self.sign_distance_field[i + 1, j] > 0: id |= 2
            if self.sign_distance_field[i, j + 1] > 0: id |= 4
            if self.sign_distance_field[i + 1, j + 1] > 0: id |= 8
            for k in ti.static(range(2)):
                if self.et[id, k][0] != -1:
                    n = ti.atomic_add(self.edge_num[None], 1)
                    self.edge[n].begin_point = self.gen_edge_pos(i, j, self.et[id, k][0])
                    self.edge[n].end_point = self.gen_edge_pos(i, j, self.et[id, k][1])
