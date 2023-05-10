from PIL import Image
from functools import reduce
import numpy as np
import time
import numbers


def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)

    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other):
        return vec3(self.x / other, self.y / other, self.z / other)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self):
        return self.dot(self)

    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def components(self):
        return (self.x, self.y, self.z)

    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))

    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(
            cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r

    def cross(self, other):
        return vec3(self.y*other.z - self.z*other.y, self.z*other.x-self.x*other.z, self.x*other.y-self.y*other.x)

    def array(self):
        return np.array([self.x, self.y, self.z])


rgb = vec3
# (w, h) = (400, 300)        # Screen size
L = vec3(5, 7, 5)        # Point light position
FARAWAY = 1.0e39           # an implausibly huge distance


def raytrace(O, D, scene, EYE, bounce=0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, scene, EYE, bounce)
            color += cc.place(hit)
    return color


class Sphere:
    def __init__(self, center, r, diffuse, mirror=0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def rotate(self, M):
        v = np.append(self.c.array(), 1)
        newCenter = M @ v
        x, y, z, w = newCenter
        self.c = vec3(x/w, y/w, z/w)

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, EYE, bounce):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (EYE - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, EYE,
                              bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color


class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker


class Triangle:
    def __init__(self, pointA: vec3, pointB: vec3, pointC: vec3, diffuse, mirror=0.5):
        self.A = pointA
        self.B = pointB
        self.C = pointC
        self.diffuse = diffuse
        self.mirror = mirror
        self.center = (self.A + self.B + self.C) / 3

    def rotate(self, M):

        vc = np.append(self.center.array(), 1)
        va = np.append(self.A.array(), 1)
        vb = np.append(self.B.array(), 1)
        vc = np.append(self.C.array(), 1)

        # Matrix mit Vektor multiplizieren
        newCenter = M @ vc
        # Variablen werden Werte aus Array zugewiesen
        x, y, z, w = newCenter
        # neuen Wert setzen - dafür durch w teilen für Umformung von homogenen zu kartesischen Koordinaten
        self.center = vec3(x/w, y/w, z/w)

        newA = M @ va
        x, y, z, w = newA
        self.A = vec3(x/w, y/w, z/w)

        newB = M @ vb
        x, y, z, w = newB
        self.B = vec3(x/w, y/w, z/w)

        newC = M @ vc
        x, y, z, w = newC
        self.C = vec3(x/w, y/w, z/w)

    def intersect(self, O, D: vec3):

        u = (self.B - self.A)
        v = (self.C - self.A)
        w = (O - self.A)
        mult = (1 / ((D.cross(v)).dot(u)))
        t = ((w.cross(u)).dot(v)) * mult
        r = ((D.cross(v)).dot(w)) * mult
        s = ((w.cross(u)).dot(D)) * mult
        pred = (((r + s) <= 1) & (r >= 0) & (r <= 1) & (s <= 1) & (s >= 0))

        return np.where(pred, t, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, EYE, bounce):
        M = (O + D * d)                         # intersection point
        N = (self.C-self.A).cross(self.B-self.A)       # normal
        toL = (L - M).norm()                    # direction to light
        toO = (EYE - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, EYE,
                              bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color


class Plane:
    def __init__(self, center, normal, diffuse, mirror=0.5):
        self.center = center
        self.normal = normal
        self.diffuse = diffuse
        self.mirror = mirror

    def rotate(self, M):

        v = np.append(self.center.array(), 1)
        newCenter = M @ v
        x, y, z, w = newCenter
        self.center = vec3(x/w, y/w, z/w)

    def intersect(self, O, D):
        N = self.normal
        OC = self.center - O
        DN = N.dot(D)
        mask = DN <= 0
        return np.where(mask, N.dot(OC) / N.dot(D), FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, EYE, bounce):
        M = (O + D * d)                         # intersection point
        N = self.normal                         # normal
        toL = (L - M).norm()                    # direction to light
        toO = (EYE - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, EYE,
                              bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color


class CheckeredPlane(Plane):
    def diffusecolor(self, M):
        checker = ((M.x).astype(int) % 2) == ((M.z).astype(int) % 2)
        return self.diffuse * checker


def test_scene(w, h, scene, EYE):
    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1, 1 / r + .25, 1, -1 / r + .25)
    x = np.tile(np.linspace(S[0], S[2], w), h)
    y = np.repeat(np.linspace(S[1], S[3], h), w)
    t0 = time.time()
    Q = vec3(x, y, 0)

    color = raytrace(EYE, (Q - EYE).norm(), scene, EYE)
    print("Took", time.time() - t0)

    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))
                            ).astype(np.uint8), "L") for c in color.components()]
    im = Image.merge("RGB", rgb)  # .save("rt3.png")
    # im.show()
    return np.array(im, dtype=np.uint8)
