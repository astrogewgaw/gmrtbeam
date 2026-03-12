from pathlib import Path
from enum import auto, Enum
from datetime import datetime
from functools import cached_property
from dataclasses import field, dataclass
from itertools import permutations, combinations

import pytz
import numpy as np
import ultraplot as uplt
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
import astropy.constants as cx
from ultraplot.axes import Axes
from joblib import delayed, Parallel
from numpy.polynomial import Polynomial
from astropy.coordinates import (
    SkyCoord,
    Latitude,
    Longitude,
    EarthLocation,
)

# GMRT's antennas list.
# NOTE: (1) The (x, y, z) coordinates are w.r.t. C02,
# and (2) C07 and S05 do not exist; however they are
# part of the antenna list for historical reasons, and
# can still be found in a lot of GMRT code as well.
GMRTANTS = {
    "C00": [6.95, 687.88, -20.04],
    "C01": [13.25, 326.45, -40.35],
    "C02": [0.00, 0.00, 0.00],
    "C03": [-51.20, -372.71, 133.59],
    "C04": [-51.01, -565.96, 123.43],
    "C05": [79.12, 67.81, -246.59],
    "C06": [71.25, -31.43, -220.58],
    "C08": [130.80, 280.68, -400.33],
    "C09": [48.61, 41.95, -151.65],
    "C10": [191.35, -164.87, -587.49],
    "C11": [102.49, -603.25, -321.56],
    "C12": [209.28, 174.85, -635.54],
    "C13": [368.67, -639.50, -1117.92],
    "C14": [207.37, -473.69, -628.63],
    "E02": [-348.18, 2814.55, 953.67],
    "E03": [-707.56, 4576.04, 1932.46],
    "E04": [-1037.59, 7780.57, 2903.29],
    "E05": [-1177.96, 10199.90, 3343.20],
    "E06": [-1572.05, 12073.32, 4543.13],
    "S01": [942.99, 633.96, -2805.93],
    "S02": [1452.91, -367.22, -4279.16],
    "S03": [2184.63, 333.10, -6404.96],
    "S04": [3072.95, 947.79, -8979.50],
    "S06": [4592.83, -369.09, -13382.48],
    "W01": [-201.35, -1591.95, 591.32],
    "W02": [-482.34, -3099.44, 1419.39],
    "W03": [-991.46, -5200.01, 2899.11],
    "W04": [-1733.91, -7039.06, 5067.53],
    "W05": [-2705.69, -8103.26, 7817.14],
    "W06": [-3101.52, -11245.77, 8916.26],
    "C07": [-3102.11, -11245.60, 8916.26],
    "S05": [-3102.11, -11245.60, 8916.26],
}

# NOTE: Names of all GMRT antennas.
GMRTANTNAMES = list(GMRTANTS.keys())

# NOTE: Total number of antennas at the GMRT.
# These are the maximum number of antennas
# that can be used in a GMRT observation.
GMRTNUMANT = len(GMRTANTS)

# NOTE: GMRT's Latitude and Longitude. These values
# are the ones provided online. Mekhala and Sanjay
# have confirmed that these are the ones used, so
# should be fine.
GMRTLOC = EarthLocation.from_geodetic(lat="19:05:47", lon="74:02:59")

# NOTE: GMRT beam polynomials for each band. The polynomials are
# as defined in AIPS PBCOR:
#   1.0
#   + X*PBPARM(3)/(10**3)
#   + X*X*PBPARM(4)/(10**7)
#   + X*X*X*PBPARM(5)/(10**10)
#   + X*X*X*X*PBPARM(6)/(10**13)
#   + X*X*X*X*X*PBPARM(7)/(10**16)
# where X = (distance from the pointing position in arc
# minutes times the frequency in GHz)**2. The coefficients
# are from Santaji's note dated 29 November 2023. We have
# replaced 1.0 with 0.5, so that the root gives the HPBW
# directly.
GMRTBMPOLYS = {
    2: Polynomial([0.5, -3.089e-3, 39.314e-7, -23.011e-10, 5.037e-13]),
    3: Polynomial([0.5, -3.129e-3, 38.816e-7, -21.608e-10, 4.483e-13]),
    4: Polynomial([0.5, -3.263e-3, 42.618e-7, -25.580e-10, 5.823e-13]),
    5: Polynomial([0.5, -2.614e-3, 27.594e-7, -13.268e-10, 2.395e-13]),
}


def xyz2uvw(xyz: np.ndarray, ha: float, dec: float, f0: float) -> np.ndarray:
    """
    Convert (x,y,z) coordinates to (u,v,w) coordinates.
    """
    return np.dot(
        np.asarray(
            [
                (np.sin(ha), np.cos(ha), 0),
                (
                    -np.sin(dec) * np.cos(ha),
                    np.sin(dec) * np.sin(ha),
                    np.cos(dec),
                ),
                (
                    np.cos(dec) * np.cos(ha),
                    -np.cos(dec) * np.sin(ha),
                    np.sin(dec),
                ),
            ]
        ),
        xyz * f0 * 1e6 / getattr(cx, "c").value,
    )


class BeamMode(Enum):
    """
    Enum describing beam modes at the GMRT.

    There are 3 beam modes at the GMRT:
        1. IA: Incoherent array, wherein voltages are added incoherently during beamforming.
               That is, I_IA = |V_1|^2 + |V_2|^2 + ... + |V_N|^2
        2. PA: Phased array, wherein voltages are added in phase during beamforming.
               That is, I_PA = |(V_1 + V_2 + ... + V_N)|^2
        3. PC: Post-correlation phased array, which can be formed on of two ways -- either
               by subtracting an IA beam from a PA beam (as is done in GMRT observations),
               or by forming the beam using visibilities (as done in SPOTLIGHT's open-sky
               and commensal modes). That is, here we get I_PC = V_1*V_2 + ... + V_N-1*V_N;
               that is, we only take the cross terms, and reject all self terms.
    """

    IA = auto()
    PA = auto()
    PC = auto()


@dataclass
class GMRTBeam:
    f0: float
    rastr: str
    decstr: str
    datestr: str
    timestr: str
    mode: BeamMode = BeamMode.PC
    data: np.ndarray | None = None
    gac: list[str] = field(default_factory=lambda: GMRTANTNAMES)

    def to_fits(self, fn: str | Path):
        if self.data is not None:
            header = self.wcs.to_header()
            hdu = fits.PrimaryHDU(header=header)
            hdu.data = self.data
            hdu.update_header()
            hdu.writeto(fn, overwrite=True)

    @cached_property
    def coords(self):
        return SkyCoord(ra=self.rastr, dec=self.decstr, frame="icrs")

    @cached_property
    def ra(self) -> Longitude:
        return getattr(self.coords, "ra")

    @cached_property
    def rarad(self) -> float:
        return float(getattr(self.ra, "radian"))

    @cached_property
    def dec(self) -> Latitude:
        return getattr(self.coords, "dec")

    @cached_property
    def decrad(self) -> float:
        return float(getattr(self.dec, "radian"))

    @cached_property
    def ist(self) -> datetime:
        dt = [self.datestr, self.timestr]
        local = pytz.timezone("Asia/Kolkata")
        return local.localize(datetime.strptime(" ".join(dt), "%d/%m/%y %H:%M:%S"))

    @cached_property
    def utc(self) -> datetime:
        return self.ist.astimezone(pytz.utc)

    @cached_property
    def time(self) -> Time:
        return Time(val=self.utc, scale="utc", location=GMRTLOC)

    @cached_property
    def lst(self) -> float:
        return float(getattr(self.time.sidereal_time(kind="mean"), "radian"))

    @cached_property
    def ha(self) -> float:
        return float(self.lst - self.rarad)

    @cached_property
    def allantennas(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            name: {
                "xyz": np.asarray(xyz),
                "uvw": xyz2uvw(
                    np.asarray(xyz),
                    f0=self.f0,
                    ha=self.ha,
                    dec=getattr(self.dec, "radian"),
                ),
            }
            for name, xyz in GMRTANTS.items()
        }

    @cached_property
    def antennas(self) -> dict[str, dict[str, np.ndarray]]:
        return {antenna: self.allantennas[antenna] for antenna in self.gac}

    @cached_property
    def antpairs(self) -> list:
        return list(combinations(self.gac, 2))

    @cached_property
    def allantpairs(self) -> list:
        return list(combinations(GMRTANTNAMES, 2))

    @cached_property
    def basepairs(self) -> list:
        return list(permutations(self.gac, 2))

    @cached_property
    def allbasepairs(self) -> list:
        return list(permutations(GMRTANTNAMES, 2))

    def baseline(self, A: str, B: str) -> float:
        uA, vA, _ = self.allantennas[A]["uvw"]
        uB, vB, _ = self.allantennas[B]["uvw"]
        uAB = uA - uB
        vAB = vA - vB
        return float(np.sqrt(uAB**2 + vAB**2))

    @cached_property
    def baselines(self) -> dict[str, float]:
        return {"-".join([A, B]): float(self.baseline(A, B)) for A, B in self.basepairs}

    @cached_property
    def allbaselines(self) -> dict[str, float]:
        return {
            "-".join([A, B]): float(self.baseline(A, B)) for A, B in self.allbasepairs
        }

    def phase(self, A: str, B: str) -> float:
        uA, vA, _ = self.allantennas[A]["uvw"]
        uB, vB, _ = self.allantennas[B]["uvw"]
        uAB = uA - uB
        vAB = vA - vB
        return float(np.arctan2(vAB, uAB))

    @cached_property
    def phases(self) -> dict[str, float]:
        return {"-".join([A, B]): float(self.phase(A, B)) for A, B in self.basepairs}

    @cached_property
    def allphases(self) -> dict[str, float]:
        return {"-".join([A, B]): float(self.phase(A, B)) for A, B in self.allbasepairs}

    @cached_property
    def size(self) -> float:
        maxbline = 0.0
        for A, B in self.antpairs:
            if maxbline < (baseline := self.baseline(A, B)):
                maxbline = baseline
        return float(1.0 / maxbline)

    @cached_property
    def fovsize(self) -> float:
        return (10 if getattr(self.dec, "dms")[0] > -20.0 else 30) * self.size

    @cached_property
    def cellsize(self) -> float:
        return self.size / 8

    @cached_property
    def npix(self) -> int:
        return int(np.floor(self.fovsize / self.cellsize))

    @cached_property
    def wcs(self) -> WCS:
        dra = ddec = np.rad2deg(self.fovsize) / self.npix
        dra = -dra
        return WCS(
            {
                "CTYPE1": "RA---SIN",
                "CTYPE2": "DEC--SIN",
                "CUNIT1": "deg",
                "CUNIT2": "deg",
                "CDELT1": dra,
                "CDELT2": ddec,
                "NAXIS1": self.npix,
                "NAXIS2": self.npix,
                "CRPIX1": self.npix / 2,
                "CRPIX2": self.npix / 2,
                "CRVAL1": self.ra.deg,
                "CRVAL2": self.dec.deg,
                "PC1_1 ": 1.0,
                "PC1_2 ": 0.0,
                "PC2_1 ": 0.0,
                "PC2_2 ": 1.0,
            }
        )

    @cached_property
    def hwhm(self) -> float:
        f0 = self.f0 / 1e3
        if 0.120 <= f0 and f0 < 0.250:
            hwhm = (np.sqrt(GMRTBMPOLYS[2].roots()[0]) / f0).real
        elif 0.250 <= f0 and f0 < 0.500:
            hwhm = (np.sqrt(GMRTBMPOLYS[3].roots()[0]) / f0).real
        elif 0.550 <= f0 and f0 < 0.850:
            hwhm = (np.sqrt(GMRTBMPOLYS[4].roots()[0]) / f0).real
        elif 1.0 <= f0 < 1.460:
            hwhm = (np.sqrt(GMRTBMPOLYS[5].roots()[0]) / f0).real
        else:
            raise ValueError(f"Frequency = {f0} MHz out of range!")
        return float(hwhm)

    @cached_property
    def uvcoverage(self):
        phz = np.asarray(list(self.phases.values()))
        bsl = np.asarray(list(self.baselines.values()))
        u = bsl * np.cos(phz)
        v = bsl * np.sin(phz)
        return u, v

    @cached_property
    def uvestimate(self):
        phz = np.asarray(list(self.phases.values()))
        bsl = np.asarray(list(self.baselines.values()))

        k = np.argmax(bsl)
        maxbsl = bsl[k]
        maxphz = phz[k]
        R0 = np.asarray([np.cos(maxphz), np.sin(maxphz)])

        minbsl = 0
        maxdot = np.cos(np.pi / 3.0)
        while minbsl == 0:
            for ii in np.arange(bsl.size):
                R = np.asarray([np.cos(phz[ii]), np.sin(phz[ii])])
                if np.abs(np.dot(R, R0)) < maxdot:
                    if minbsl < bsl[ii]:
                        minbsl = bsl[ii]
            maxdot = maxdot + 0.05
        minbsl = 0.1 * maxbsl if minbsl < 0.1 * maxbsl else minbsl

        x = np.zeros(360)
        y = np.zeros(360)
        angles = np.linspace(0, 2 * np.pi, 360)
        for ii in np.arange(angles.size):
            angle = angles[ii]
            tx = maxbsl * np.cos(angle)
            ty = minbsl * np.sin(angle)
            x[ii] = tx * np.cos(maxphz) - ty * np.sin(maxphz)
            y[ii] = ty * np.cos(maxphz) + tx * np.sin(maxphz)
        return x, y

    def compute(self):
        beamgrid = np.ndarray((self.npix, self.npix), dtype=np.float32)

        def _(i):
            ll = (-0.5 * self.fovsize) + (i * (self.fovsize / self.npix))
            for j in np.arange(self.npix):
                mm = (-0.5 * self.fovsize) + (j * (self.fovsize / self.npix))
                nn = np.sqrt(1 - ll**2 - mm**2)
                match self.mode:
                    case BeamMode.PA:
                        z1 = 0.0
                        z2 = 0.0
                        for A, B in self.antpairs:
                            uA, vA, wA = self.antennas[A]["uvw"]
                            uB, vB, wB = self.antennas[B]["uvw"]
                            X = (uA - uB) * ll + (vA - vB) * mm + (wA - wB) * (nn - 1)
                            z1 += np.cos(2 * np.pi * X)
                            z2 += np.sin(2 * np.pi * X)
                        beamgrid[j][i] = np.sqrt(z1**2 + z2**2)
                    case BeamMode.PC:
                        z = 0
                        for A, B in self.antpairs:
                            uA, vA, wA = self.antennas[A]["uvw"]
                            uB, vB, wB = self.antennas[B]["uvw"]
                            X = (uA - uB) * ll + (vA - vB) * mm + (wA - wB) * (nn - 1)
                            z = z + np.cos(2 * np.pi * X)
                        beamgrid[j][i] = z

        Parallel(
            njobs=-1,
            verbose=0,
            return_as="list",
            require="sharedmem",
        )(delayed(_)(i) for i in np.arange(self.npix))

        beamgrid /= beamgrid.max()
        self.data = beamgrid

    def plot(
        self,
        ax=None,
        show: bool = True,
        save: str | None = None,
        **kwargs,
    ):
        def plotter(ax):
            if self.data is not None:
                side = np.rad2deg(self.fovsize) * 3600
                l0, l1 = -side / 2, side / 2
                m0, m1 = -side / 2, side / 2
                hm = ax.imshow(
                    self.data,
                    cmap="batlow",
                    origin="lower",
                    vmin=self.data.min(),
                    vmax=self.data.max(),
                    extent=(l0, l1, m0, m1),
                )
                ax.invert_xaxis()
                ax.format(
                    title="Synthesized beam",
                    xlabel="l (East-West) arcsec",
                    ylabel="m (North-South) arcsec",
                )
                ax.colorbar(hm)

        if ax is None:
            if self.data is not None:
                fig = getattr(uplt, "figure")(width=3.5, height=3.5)
                plotter(fig.subplot())
                if show:
                    getattr(uplt, "show")()
                if save is not None:
                    fig.savefig(save, dpi=kwargs.get("dpi", 150))
        else:
            plotter(ax)

    def plotuv(
        self,
        ax=None,
        show: bool = True,
        save: str | None = None,
        **kwargs,
    ):
        def plotter(ax):
            if self.data is not None:
                ax.scatter(*self.uvcoverage, edgecolor="black")
                ax.plot(*self.uvestimate, lw=2, color="red")
                ax.format(
                    xlabel="u",
                    ylabel="v",
                    title=r"$uv$-coverage",
                )

        if ax is None:
            fig = getattr(uplt, "figure")(width=3.5, height=3.5)
            plotter(fig.subplot())
            if show:
                getattr(uplt, "show")()
            if save is not None:
                fig.savefig(save, dpi=kwargs.get("dpi", 150))
        else:
            plotter(ax)

    def plotsky(
        self,
        ax: Axes | None = None,
        show: bool = True,
        save: str | None = None,
        **kwargs,
    ):
        def plotter(ax):
            if self.data is not None:
                hm = ax.imshow(
                    self.data,
                    cmap="batlow",
                    origin="lower",
                    vmin=self.data.min(),
                    vmax=self.data.max(),
                )
                ax.colorbar(hm)
                ax.invert_xaxis()
                ax.set_title("Synthesized beam")
                ax.set_ylabel("Declination (Dec)")
                ax.set_xlabel("Right ascension (RA)")

        if ax is None:
            if self.data is not None:
                fig = getattr(uplt, "figure")(width=4, height=4)
                plotter(fig.subplot(projection=self.wcs))
                if show:
                    getattr(uplt, "show")()
                if save is not None:
                    fig.savefig(save, dpi=kwargs.get("dpi", 150))
        else:
            plotter(ax)
