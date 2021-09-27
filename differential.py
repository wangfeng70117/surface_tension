import taichi as ti

ti.init(arch=ti.gpu)


@ti.data_oriented
class Differential:
    def __init__(self,
                 n_grid = 128,
                 min_x = .0,
                 max_x = 1.0,
                 field = None,
                 dim = 2,
                 gradient = None,
                 divergence = None,
                 laplacian = None):
        self.n_grid = n_grid
        self.max_x = max_x
        self.minx_x = min_x
        self.field = field
        self.dim = dim
        self.dx = (max_x - min_x) / n_grid
        self.inv_dx = 1 / self.dx
        self.gradient = gradient
        self.divergence = divergence
        self.laplacian = laplacian
        # 球星level set 的半径
        self.radius = self.dx * 0.5
    @ti.func
    def ti_vector2(self, u, v):
        return ti.Vector([u, v])

    # 中心差分计算梯度场
    @ti.kernel
    def central_difference(self):
        for I in ti.grouped(self.field):
            i, j = I
            u, v = .0, .0
            # 区分边界位置
            if i == 0:
                u = (self.field[i + 1, j] - self.field[i, j]) * 0.5 * self.inv_dx
            elif i == self.n_grid - 1:
                u = (self.field[i, j] - self.field[i - 1, j]) * 0.5 * self.inv_dx
            else:
                u = (self.field[i + 1, j] - self.field[i - 1, j]) * 0.5 * self.inv_dx

            if j == 0:
                v = (self.field[i, j + 1] - self.field[i, j]) * 0.5 * self.inv_dx
            elif j == self.n_grid - 1:
                v = (self.field[i, j] - self.field[i, j - 1]) * 0.5 * self.inv_dx
            else:
                v = (self.field[i, j + 1] - self.field[i, j - 1]) * 0.5 * self.inv_dx

            self.gradient[I] = self.ti_vector2(u, v)

    # 中心差分计算散度场
    @ti.kernel
    def central_divergence(self):
        for I in ti.grouped(self.field):
            i, j = I
            u, v = .0, .0
            if i == 0:
                u = (self.gradient[i + 1, j].x - self.gradient[i, j].x) * 0.5 * self.inv_dx
            elif i == self.n_grid - 1:
                u = (self.gradient[i, j].x - self.gradient[i - 1, j].x) * 0.5 * self.inv_dx
            else:
                u = (self.gradient[i + 1, j].x - self.gradient[i - 1, j].x) * 0.5 * self.inv_dx

            if j == 0:
                v = (self.gradient[i, j + 1].y - self.gradient[i, j].y) * 0.5 * self.inv_dx
            elif j == self.n_grid - 1:
                v = (self.gradient[i, j].y - self.gradient[i, j - 1].y) * 0.5 * self.inv_dx
            else:
                v = (self.gradient[i, j + 1].y - self.gradient[i, j - 1].y) * 0.5 * self.inv_dx

            self.divergence[I] = u + v

    # 中心差分计算拉普拉斯算子
    @ti.kernel
    def central_laplacian(self):
        for I in ti.grouped(self.field):
            i, j = I
            u, v = .0, .0
            if i == 0:
                u = (self.field[i + 1, j] - 2 * self.field[i, j] + self.field[i, j]) * self.inv_dx * self.inv_dx
            elif i == self.n_grid - 1:
                u = (self.field[i, j] - 2 * self.field[i, j] + self.field[i - 1, j]) * self.inv_dx * self.inv_dx
            else:
                u = (self.field[i + 1, j] - 2 * self.field[i, j] + self.field[i - 1, j]) * self.inv_dx * self.inv_dx

            if j == 0:
                v = (self.field[i, j + 1] - 2 * self.field[i, j] + self.field[i, j]) * self.inv_dx * self.inv_dx
            elif j == self.n_grid - 1:
                v = (self.field[i, j] - 2 * self.field[i, j] + self.field[i, j - 1]) * self.inv_dx * self.inv_dx
            else:
                v = (self.field[i, j + 1] - 2 * self.field[i, j] + self.field[i, j - 1]) * self.inv_dx * self.inv_dx

            self.laplacian[I] = u + v



