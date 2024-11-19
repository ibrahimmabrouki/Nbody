Skip to content
Navigation Menu
hamdanabdellatef
/
Nbody

Type / to search
Code
Issues
Pull requests
Actions
Projects
Security
Insights
 The password you provided is in a list of passwords commonly used on other websites. To increase your security, you must update your password. After December 19, 2024 we will automatically reset your password. Change your password on the settings page.

Read our documentation on safer password practices.

Comparing changes
Choose two branches to see what’s changed or to start a new pull request. If you need to, you can also  or learn more about diff comparisons.
 
 
...
 
 
  Able to merge. These branches can be automatically merged.
Discuss and review the changes in this comparison with others. Learn about pull requests
 1 commit
 1 file changed
 1 contributor
Commits on Nov 19, 2024
Create optimizednbody.cu

@ibrahimmabrouki
ibrahimmabrouki authored 10 minutes ago
 Showing  with 118 additions and 0 deletions.
 118 changes: 118 additions & 0 deletions118  
optimizednbody.cu
Original file line number	Original file line	Diff line number	Diff line change
@@ -0,0 +1,118 @@
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f
#define BLOCK_SIZE 256  // Number of threads per block

typedef struct { float x, y, z, vx, vy, vz; } Body;

/* Device kernel to calculate gravitational forces between bodies*/

_global_ void bodyForceKernel(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // Ensure thread doesn't access out-of-bounds memory

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < n; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
}

/* Device kernel to integrate positions of bodies.*/

_global_ void integratePositionsKernel(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // Ensure thread doesn't access out-of-bounds memory

    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
}

int main(const int argc, const char** argv) {
    int nBodies = 2 << 11;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    const char *initialized_values;
    const char *solution_values;

    if (nBodies == 2 << 11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else {
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f;  // Time step
    const int nIters = 10;   // Simulation iterations

    int bytes = nBodies * sizeof(Body);
    Body p = (Body)malloc(bytes);
    Body *d_p;

    // Read initial values from file
    read_values_from_file(initialized_values, (float*)p, bytes);

    // Allocate memory on the GPU
    cudaMalloc(&d_p, bytes);

    // Copy initial values to GPU
    cudaMemcpy(d_p, p, bytes, cudaMemcpyHostToDevice);

    double totalTime = 0.0;

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        // Launch the bodyForce kernel
        int blocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bodyForceKernel<<<blocks, BLOCK_SIZE>>>(d_p, dt, nBodies);
        cudaDeviceSynchronize();

        // Launch the integratePositions kernel
        integratePositionsKernel<<<blocks, BLOCK_SIZE>>>(d_p, dt, nBodies);
        cudaDeviceSynchronize();

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    double avgTime = totalTime / (double)nIters;
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

    // Copy results back to CPU
    cudaMemcpy(p, d_p, bytes, cudaMemcpyDeviceToHost);

    // Write results to file
    write_values_to_file(solution_values, (float*)p, bytes);

    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    // Free allocated memory
    free(p);
    cudaFree(d_p);

    return 0;
}
Footer
© 2024 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact
Manage cookies
Do not share my personal information
