from rendering import Scene, RenderWindow
import numpy as np
from rt3 import *


class RayTracer:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.scene = [
            Sphere(vec3(.5, .1, 1), .4, vec3(1, 0, 0)),
            Sphere(vec3(0, .9, 1), .4, vec3(0, 0, 1)),
            Sphere(vec3(-.5, .1, 1), .4, vec3(0, 1, 0)),
            CheckeredPlane(vec3(-2.75, -3, 3.5), vec3(0, 1, 0), vec3(1, 1, 1)),
            #Triangle(vec3(.5, .1, 1), vec3(0, .9, 1), vec3(-.5, .1, 1))
            #CheckeredSphere(vec3(0, -9.5, 0), 9, vec3(.75, .75, .75), 0.25),
        ]
        self.camera = Camera(vec3(0, 0, 0), vec3(
            0, 1, 0), vec3(0, 0, 1), np.pi/3, 1)

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
        print('Taste wurde gedrückt')

        # TODO: modify scene accordingly
        pass

    def rotate_neg(self):
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
