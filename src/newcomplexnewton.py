import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from numba import cuda
import math
import time
import cmath

uf = "((z**(3))-(1))"
udf = "3*z**2"

funcf = f"def f(z):\n return {uf}"
funcdf = f"def df(z):\n return {udf}"

namespace = {"cmath": cmath}
exec(funcf, namespace)
exec(funcdf, namespace)

ff = namespace["f"]
fdf = namespace["df"]

def kernelFactory(func, dfunc):
    f = cuda.jit(device=True)(func)
    df = cuda.jit(Device=True)(dfunc)

    @cuda.jit
    def kernel(RResult, IResult, brightness, xMin:int, xMax:int, yMin:int, yMax:int, maxIterations:int, convergenceDelta:float, alpha:float, beta:float):
        xIdx, yIdx = cuda.grid(2)
            
        height, width = RResult.shape
        if (xIdx >= width) or (yIdx >= height): return
            
        z_r = xMin + (xIdx/width) * (xMax-xMin)
        z_i = yMin + (yIdx/height) * (yMax-yMin)
        z = complex(z_r, z_i)

        for i in range(maxIterations):
            f_z = f(z)
            df_z = df(z)

            if abs(df_z) == 0: break

            z = z - f_z/df_z
                
            absDiff = abs(f_z/df_z)
            if absDiff < convergenceDelta:
                RResult[yIdx, xIdx] = z.real
                IResult[yIdx, xIdx] = z.imag
                    
                if absDiff < 1e-15: absDiff = 1e-15
                
                smoothVal = i - math.log2(-math.log(absDiff)) + alpha/2
                    
                brightnessValue = smoothVal * beta
                if brightnessValue > 1.0: brightnessValue = 1.0
                if brightnessValue < 0.0: brightnessValue = 0.0
                    
                brightness[yIdx, xIdx] = brightnessValue
                return

        RResult[yIdx, xIdx] = 0.0
        IResult[yIdx, xIdx] = 0.0
        brightness[yIdx, xIdx] = 0.0

    return kernel

class ComplexNewton():
    def __init__(self, xMin:int=-8, xMax:int=8, yMin:int=-8, yMax:int=8, numPointsPerAxis:int=3000, convergenceDelta:float=1e-6, maxIterations:int=100, alpha:float=3.0, beta:float=0.05, maxColorOptions:int=20) -> None:
        self.xMin  = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.numPointsPerAxis = numPointsPerAxis
        self.convergenceDelta = convergenceDelta
        self.maxIterations = maxIterations
        self.alpha = alpha 
        self.beta = beta
        self.maxColorOptions = maxColorOptions
    

    def run(self) -> float:
        startMilli = int(time.time()*1000)

        RRoots = np.zeros((self.numPointsPerAxis, self.numPointsPerAxis), dtype=np.float64)
        IRoots = np.zeros((self.numPointsPerAxis, self.numPointsPerAxis), dtype=np.float64)
        brightness = np.zeros((self.numPointsPerAxis, self.numPointsPerAxis), dtype=np.float64)

        DRRoots = cuda.to_device(RRoots)
        DIRoots = cuda.to_device(IRoots)
        DBrightness = cuda.to_device(brightness)

        threadsPerBlock = (16, 16)
        blocksX = int(np.ceil(self.numPointsPerAxis / threadsPerBlock[0]))
        blocksY = int(np.ceil(self.numPointsPerAxis / threadsPerBlock[1]))
        grid = (blocksX, blocksY)

        kernel = kernelFactory(ff, fdf)

        kernel[grid, threadsPerBlock](DRRoots, DIRoots, DBrightness, self.xMin, self.xMax, self.yMin, self.yMax, self.maxIterations, self.convergenceDelta, self.alpha, self.beta)
        cuda.synchronize()

        RRoots = DRRoots.copy_to_host()
        IRoots = DIRoots.copy_to_host()
        brightness = DBrightness.copy_to_host()

        Z = RRoots + 1j*IRoots

        pointsConverged = (brightness > 0)

        hsv = np.zeros((self.numPointsPerAxis, self.numPointsPerAxis, 3))
        angles = np.angle(Z)

        hsv[pointsConverged, 0] = (angles[pointsConverged] + np.pi) / (2*np.pi)
        hsv[pointsConverged, 1] = 1.0
        hsv[pointsConverged, 2] = brightness[pointsConverged]

        rgb = mcolors.hsv_to_rgb(hsv)

        fig, ax = plt.subplots(figsize=(8,8), dpi=100)
        ax.imshow(rgb, extent=[self.xMin, self.xMax, self.yMin, self.yMax], origin="lower", interpolation="antialiased")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Re(z)")
        ax.set_ylabel('Im(z)')
        ax.set_title(f"Newton-Raphson convergence in the complex plane based on $f(z)=z^4+z^2+1$ with $\\alpha={self.alpha}$,\n$\\beta={self.beta}$, $N={self.numPointsPerAxis}$.")
        ax.set_aspect("equal", adjustable="box")

        plt.savefig("GPU_2.png")
        endMilli = int(time.time()*1000)

        return (endMilli-startMilli)/1000

if __name__ == "__main__":
    cn = ComplexNewton()
    runtime = cn.run()
    print(runtime)