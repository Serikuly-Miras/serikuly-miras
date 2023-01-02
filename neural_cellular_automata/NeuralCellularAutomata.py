import taichi as ti
import numpy as np


@ti.data_oriented
class NeuralCellularAutomata:
    def __init__(self, vp: tuple, scale_factor: int) -> None:
        # Innit taichi on gpu
        ti.init(arch=ti.cpu, dynamic_index=True)

        self.vp = vp
        self.scale_factor = scale_factor
        self.grid_size = (vp[0] // scale_factor, vp[1] // scale_factor)

        self.pixels = ti.Vector.field(n=3, dtype=float, shape=(vp))
        self.grid = ti.field(dtype=float, shape=self.grid_size)
        self.grid_buffer = ti.field(dtype=float, shape=self.grid_size)

        self.core = ti.Vector.field(n=3, dtype=float, shape=3)
        self.core[0].xyz = [0.68, -0.9, 0.68]
        self.core[1].xyz = [-0.9, -0.66, -0.9]
        self.core[2].xyz = [0.68, -0.9, 0.68]

        self.fill_random()

    def record(self, frames: int, fps: int) -> None:
        result_dir = "./results"
        video_manager = ti.tools.VideoManager(
            output_dir=result_dir, framerate=fps, automatic_build=False)

        for i in range(frames):
            self.update_grid()
            self.update_grid()
            self.paint()

            video_manager.write_frame(self.pixels.to_numpy())
            print(f'\rFrame {i+1}/{frames} is recorded', end='')

        print()
        print('Exporting .gif ...')
        video_manager.make_video(gif=True)
        print(
            f'GIF video is saved to {video_manager.get_output_filename(".gif")}')

    def draw(self):
        gui = ti.GUI("NCA", res=(self.vp), fast_gui=True)
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            # update twice show once (skip frame)
            self.update_grid()
            self.update_grid()

            self.paint()
            gui.set_image(self.pixels)
            gui.show()

    @ti.kernel
    def fill_random(self):
        for i, j in self.grid:
            self.grid[i, j] = ti.random(ti.f32)

    @ti.kernel
    def update_grid(self):
        w = self.grid_size[0]
        h = self.grid_size[1]
        for i, j in self.grid:
            res = 0.0
            for l in range(3):
                for n in range(3):
                    r = i + l - 1
                    c = j + n - 1

                    if r < 0:
                        r += w
                    elif r > w:
                        r -= w

                    if c < 0:
                        c += h
                    elif c > h:
                        c -= h

                    res += self.grid[r, c] * self.core[l][n]
            self.grid_buffer[i, j] = self.activation(res)

        for i, j in self.grid:
            self.grid[i, j] = ti.math.clamp(self.grid_buffer[i, j], 0, 1)

    @ti.kernel
    def paint(self):
        for i, j in self.pixels:
            val = self.grid[i // self.scale_factor, j // self.scale_factor]
            self.pixels[i, j].r = val
            self.pixels[i, j].g = val
            self.pixels[i, j].b = val

    @ti.func
    def activation(self, x: float) -> float:
        return -1.0/ti.pow(2.0, (0.6*ti.pow(x, 2.0)))+1.
