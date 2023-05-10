from rendering import Scene, RenderWindow
import numpy as np
from rt3 import *


class RayTracer:

    def __init__(self, width, height):
        self.EYE = vec3(0, 0, 5)      # Eye position
        self.UP = vec3(0, 1, 0)
        self.center = vec3(0, 0, -5)
        self.width = width
        self.height = height
        self.scene = [
            Sphere(vec3(.5, .1, -5), .4, vec3(1, 0, 0)),
            Sphere(vec3(0, .9, -5), .4, vec3(0, 0, 1)),
            Sphere(vec3(-.5, .1, -5), .4, vec3(0, 1, 0)),
            CheckeredPlane(vec3(0, -1, 0), vec3(0, 1, 0), vec3(1, 1, 1)),
            #Triangle(vec3(.5, .1, 1), vec3(0, .9, 1), vec3(-.5, .1, 1))
        ]

        self.rendered = test_scene(
            self.width, self.height, self.scene, self.EYE)

        # TODO: setup your ray tracer

    def resize(self, new_width, new_height):
        self.width = new_width
        self.height = new_height
        self.rendered = test_scene(
            self.width, self.height, self.scene, self.EYE)

    def rotate_pos(self):
        alpha = -np.pi/10
        c = np.cos(alpha)
        s = np.sin(alpha)

        C = np.array([[1, 0, 0, -self.center.x], [0, 1, 0, -
                     self.center.y], [0, 0, 1, -self.center.z], [0, 0, 0, 1]])
        R = np.array([[c, 0, -s, 0], [0, 1, 0, 0],
                      [s, 0, c, 0], [0, 0, 0, 1]])
        CI = np.array([[1, 0, 0, self.center.x], [0, 1, 0, self.center.y], [
                      0, 0, 1, self.center.z], [0, 0, 0, 1]])
        M = CI @ R @ C
        for e in self.scene:
            e.rotate(M)

        self.rendered = test_scene(
            self.width, self.height, self.scene, self.EYE)
        pass

    def rotate_neg(self):
        # die Winkel sind bewusst vertauscht, da die Kameraposition gespiegelt ist
        alpha = np.pi/10
        c = np.cos(alpha)
        s = np.sin(alpha)

        C = np.array([[1, 0, 0, -self.center.x], [0, 1, 0, -
                     self.center.y], [0, 0, 1, -self.center.z], [0, 0, 0, 1]])
        R = np.array([[c, 0, -s, 0], [0, 1, 0, 0],
                      [s, 0, c, 0], [0, 0, 0, 1]])
        CI = np.array([[1, 0, 0, self.center.x], [0, 1, 0, self.center.y], [
                      0, 0, 1, self.center.z], [0, 0, 0, 1]])
        M = CI @ R @ C
        for e in self.scene:
            e.rotate(M)

        self.rendered = test_scene(
            self.width, self.height, self.scene, self.EYE)
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
