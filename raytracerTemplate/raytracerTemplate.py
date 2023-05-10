from rendering import Scene, RenderWindow
import numpy as np
from rt3 import *


class RayTracer:

    def __init__(self, width, height):
        self.UP = vec3(0, 1, 0)
        self.width = width
        self.height = height
        self.scene = [
            Sphere(vec3(.5, .1, 0), .4, vec3(1, 0, 0)),
            Sphere(vec3(0, .9, 0), .4, vec3(0, 0, 1)),
            Sphere(vec3(-.5, .1, 0), .4, vec3(0, 1, 0)),
            CheckeredPlane(vec3(-2.75, -3, 3.5), vec3(0, 1, 0), vec3(1, 1, 1)),
            #Triangle(vec3(.5, .1, 1), vec3(0, .9, 1), vec3(-.5, .1, 1))
        ]

        self.rendered = test_scene(
            self.width, self.height, self.scene, self.camera)

        # TODO: setup your ray tracer

    def resize(self, new_width, new_height):
        self.width = new_width
        self.height = new_height
        self.rendered = test_scene(
            self.width, self.height, self.scene, self.camera)

        # TODO: modify scene accordingly

    def rotate_pos(self):
        alpha = np.pi/3
        c = np.cos(alpha)
        s = np.sin(alpha)

        R = np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # TODO: modify scene accordingly
        pass

    def rotate_neg(self):
        alpha = -np.pi/3
        print('Taste wurde gedrückt')

        # TODO: modify scene accordingly
        pass

    def render(self):
        # TODO: Replace Dummy Data with Ray Traced Data
        return self.rendered


# main function
if __name__ == '__main__':

    # set size of render viewport
    width, height = 400, 300

    # instantiate a ray tracer
    ray_tracer = RayTracer(width, height)

    # instantiate a scene
    scene = Scene(width, height, ray_tracer, "Raytracing Template")

    # pass the scene to a render window
    rw = RenderWindow(scene)

    # ... and start main loop
    rw.run()
