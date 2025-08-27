from dataclasses import dataclass

import numpy as np
import astropy.units as ux
from astropy.coordinates import SkyCoord

from gmrtbeam.fit import BeamEllipse


@dataclass
class GMRTMosaic:
    nbeams: int
    ellipse: BeamEllipse
    coords: SkyCoord | None = None
    grid: tuple[np.ndarray, np.ndarray] | None = None

    def _gridder(self, R: float, recenter: bool = True):
        N = self.nbeams
        t = self.ellipse.t
        t = np.arctan2(np.sin(t), -np.cos(t))
        N2 = 4 * np.floor(R / self.ellipse.b).astype(int)
        N1 = 4 * np.floor(R / (np.sqrt(3) / 2 * self.ellipse.a)).astype(int)
        X, Y = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)

        dx, dy = np.sqrt(3) / 2 * self.ellipse.a, self.ellipse.b
        x = -R - 0.5 * dx
        i = 0
        for ii in range(N1):
            y = -R - 0.5 * dy
            y = y + 0.5 * dy if ii % 2 else y
            for _ in range(N2):
                r = np.sqrt(x * x + y * y)
                if r < R:
                    X[i], Y[i] = x, y
                    i = i + 1
                    if i == N:
                        break
                y = y + dy
            x = x + dx
            if i == N:
                break
        XPOLAR = X * np.cos(t) - Y * np.sin(t)
        YPOLAR = X * np.sin(t) + Y * np.cos(t)
        if recenter:
            radii = np.sqrt(XPOLAR[:i] * XPOLAR[:i] + YPOLAR[:i] * YPOLAR[:i])
            icenter = np.argmin(radii)
            X0, Y0 = XPOLAR[icenter], YPOLAR[icenter]
            XPOLAR = XPOLAR - X0
            YPOLAR = YPOLAR - Y0
            return (XPOLAR, YPOLAR), i, (-X0, -Y0)
        return (XPOLAR, YPOLAR), i, (0, 0)

    def tile(
        self,
        eff: float = 0.9,
        fac: float = 1.05,
        dgrid0: float = 0.05,
        maxiterin: int = 50,
        maxiterout: int = 10,
        beamusage: float = 0.95,
        areausage: float = 0.85,
        recenter: bool = True,
    ):
        N = self.nbeams
        X0, Y0 = 0.0, 0.0
        R = 0.5 * np.sqrt((N * self.ellipse.a * self.ellipse.b) / eff)
        x, y = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)

        i = 0
        Nopt = 0
        dgrid = dgrid0
        increase = False
        outconverge = False
        while not outconverge:
            j = 0
            nt0 = 0
            Ropt = R
            ntmax = 0
            inconverge = False
            n = np.floor(N * fac).astype(int)
            while not inconverge:
                (x, y), Nopt, (X0, Y0) = self._gridder(R, recenter=recenter)
                farea = Nopt * (self.ellipse.a * self.ellipse.b) / (4 * R * R)
                if Nopt == n:
                    if np.sqrt(farea) > 0.95:
                        R = 0.95 * R
                    else:
                        dgrid = dgrid0
                        R = np.sqrt(farea) * R
                else:
                    if farea > areausage:
                        if Nopt > ntmax:
                            Ropt = R
                            ntmax = Nopt
                    if Nopt >= nt0:
                        if (farea > areausage) and (Nopt >= beamusage * n):
                            inconverge = True
                            break
                    if increase and (Nopt > 0.7 * n):
                        dgrid = 0.9 * dgrid
                    else:
                        increase = True
                        dgrid = dgrid0
                        R = (1 + dgrid) * R
                nt0 = Nopt
                j = j + 1
                if j > maxiterin:
                    break
            if not inconverge:
                R = Ropt
                (x, y), Nopt, (X0, Y0) = self._gridder(R, recenter=recenter)
            if Nopt < N:
                fac = fac + 0.05
            else:
                outconverge = True
                break
            i = i + 1
            if i > maxiterout:
                break
        if Nopt > N:
            D = np.sqrt((x[:Nopt] - X0) ** 2 + (y[:Nopt] - Y0) ** 2)
            distant = (np.argsort(D)[N:Nopt]).tolist()
            X, Y = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)

            i = 0
            for ix in range(Nopt):
                if ix not in distant:
                    X[i], Y[i] = x[i], y[i]
                    i = i + 1
            Nopt = N
        else:
            X, Y = x, y
        farea = Nopt * (self.ellipse.a * self.ellipse.b) / (4 * R * R)
        return (X, Y), R, Nopt, (X0, Y0)
