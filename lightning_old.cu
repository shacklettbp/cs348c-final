#include <iostream>
#include <cuda_fp16.h>
#include <math.h>
#include <cooperative_groups.h>
#include <unistd.h>
#include <random>
#include <cassert>
#include <vector>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

const unsigned GRID_SZ = 128;
const float h = 1.0/(GRID_SZ + 1.0);
const float h2 = h*h;

inline __host__ __device__ 
int get_idx(int x, int y, int z)
{
  const int outer_sz = GRID_SZ + 2;
  z += 1;
  y += 1;
  x += 1;
  return x + outer_sz * (y + outer_sz * z);
}

inline __host__ __device__
int get_idx(dim3 pos)
{
  return get_idx(pos.x, pos.y, pos.z);
}

// Computes LHS of \del^2(\phi) = 4*\pi*\rho
__device__
void compute_error(half *phi_estimate, half *rho, half *err, dim3 pos) 
{
  half cur = phi_estimate[get_idx(pos)];
  half h2h = __float2half(h2);

  half delx = h2h * (phi_estimate[get_idx(pos.x - 1, pos.y, pos.z)] + phi_estimate[get_idx(pos.x + 1, pos.y, pos.z)] - __float2half(2.0)*cur);

  half dely = h2h * (phi_estimate[get_idx(pos.x, pos.y - 1, pos.z)] + phi_estimate[get_idx(pos.x, pos.y + 1, pos.z)] - __float2half(2.0)*cur);

  half delz = h2h * (phi_estimate[get_idx(pos.x, pos.y, pos.z - 1)] + phi_estimate[get_idx(pos.x, pos.y, pos.z + 1)] - __float2half(2.0)*cur);

  err[get_idx(pos)] = __float2half(abs(__half2float(delx + dely + delz + __float2half(4*M_PI)*rho[get_idx(pos)])));
}

inline __device__
void relax_iter(half *cur_state, half *new_phi, half *old_phi, half *rho, dim3 pos)
{
  half cur_fixed = cur_state[get_idx(pos)];
  if (cur_fixed != __float2half(-1.0)) {
    // Boundary condition
    new_phi[get_idx(pos)] = cur_fixed;
  } else {
    half cur_val = old_phi[get_idx(pos)];
    half cur_rho = __float2half(-4.0*M_PI)*rho[get_idx(pos)];

    half s = 1.0;

    half new_val = s * __float2half(1.0/6.0) *
      (
       old_phi[get_idx(pos.x + 1, pos.y, pos.z)] +
       old_phi[get_idx(pos.x - 1, pos.y, pos.z)] +
       old_phi[get_idx(pos.x, pos.y + 1, pos.z)] +
       old_phi[get_idx(pos.x, pos.y - 1, pos.z)] +
       old_phi[get_idx(pos.x, pos.y, pos.z + 1)] +
       old_phi[get_idx(pos.x, pos.y, pos.z - 1)] -
       __float2half(h2) * cur_rho
      ) + (__float2half(1.0) - s)*cur_val;

    new_phi[get_idx(pos)] = new_val;
  }
}

__global__
void calculate_phi(half *cur_state, half *phi, half *tmp, half *rho)
{
  dim3 pos((blockIdx.x * blockDim.x) + threadIdx.x,
           (blockIdx.y * blockDim.y) + threadIdx.y,
           (blockIdx.z * blockDim.z) + threadIdx.z);

  auto grid = cooperative_groups::this_grid();

  const int phi_iters = 1000;
  for (int i = 0; i < phi_iters; i++) {
    tmp[get_idx(pos)] = phi[get_idx(pos)];
    grid.sync();
    relax_iter(cur_state, phi, tmp, rho, pos);
  }
  compute_error(phi, rho, tmp, pos);
}

__global__ 
void sum_arr(half *grid, float *sum)
{
  uint x = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint y = (blockIdx.y * blockDim.y) + threadIdx.y;
  uint z = (blockIdx.z * blockDim.z) + threadIdx.z;

  float cur = __half2float(grid[get_idx(x, y, z)]);
  atomicAdd(sum, cur);
}

struct candidate {
  int x, y, z;
  float unnorm_prob;
};

void find_elem(half *phi, half *state, float *rand) {
  float prob_sum = 0;
  std::vector<candidate> candidates;
  for (int i = 0; i < GRID_SZ; i++) {
    for (int j = 0; j < GRID_SZ; j++) {
      for (int k = 0; k < GRID_SZ; k++) {
        unsigned idx = get_idx(i, j, k);
        if (__half2float(state[idx]) == 0.0) {
          continue;
        }
        if (j != GRID_SZ/2) { // constrain to 2d
          continue;
        }
        if (__half2float(phi[get_idx(i - 1, j, k)]) == 0.0 ||
            __half2float(phi[get_idx(i + 1, j, k)]) == 0.0 ||
            __half2float(phi[get_idx(i, j - 1, k)]) == 0.0 ||
            __half2float(phi[get_idx(i, j + 1, k)]) == 0.0 ||
            __half2float(phi[get_idx(i, j, k - 1)]) == 0.0 ||
            __half2float(phi[get_idx(i, j, k + 1)]) == 0.0) {
          float val = __half2float(phi[idx]);
          float unnorm_prob = powf(val, 3);
          candidates.emplace_back(candidate {i, j, k, unnorm_prob});
          prob_sum += unnorm_prob;
        }
      }
    }
  }

  //int idx = *rand * candidates.size();
  //auto &c = candidates[idx];
  //state[get_idx(c.x, c.y, c.z)] = __float2half(0);
  //if (c.z == GRID_SZ - 1) {
  //  *rand = -1.0;
  //}

  float cum = 0;
  for (auto &c : candidates) {
    cum += c.unnorm_prob / prob_sum;
    if (*rand < cum) {
       state[get_idx(c.x, c.y, c.z)] = __float2half(0);
       if (c.z == GRID_SZ - 1) {
         *rand = -1.0;
       }
       return;
    }
  }
  
  std::cout << *rand << " " << cum << std::endl;
  assert(false);
}

int main(void)
{
  std::random_device rd;  
  std::mt19937 gen(rd()); 
  std::uniform_real_distribution<> dis(0, 1.0);

  cudaSetDevice(0);
  size_t free_bytes, total_bytes;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  //printf("Free bytes on device: %lu\n", free_bytes);

  unsigned num_grid_elems = (GRID_SZ+2)*(GRID_SZ+2)*(GRID_SZ+2);
  unsigned grid_storage_bytes = num_grid_elems*sizeof(half);

  half *cur_state;
  half *phi;
  half *scratch;
  half *rho;

  cudaMallocManaged(&cur_state, grid_storage_bytes);
  cudaCheckError();
  cudaMallocManaged(&phi, grid_storage_bytes);
  cudaCheckError();
  cudaMallocManaged(&scratch, grid_storage_bytes);
  cudaCheckError();
  cudaMallocManaged(&rho, grid_storage_bytes);
  cudaCheckError();

  cudaMemset(cur_state, 0, grid_storage_bytes);
  cudaCheckError();
  cudaMemset(phi, 0, grid_storage_bytes);
  cudaCheckError();
  cudaMemset(scratch, 0, grid_storage_bytes);
  cudaCheckError();
  cudaMemset(rho, 0, grid_storage_bytes);
  cudaCheckError();

  dim3 threadsPerBlock(32, 32, 1);
  dim3 numBlocks(GRID_SZ / threadsPerBlock.x, GRID_SZ / threadsPerBlock.y, GRID_SZ / threadsPerBlock.z);

  // Initialize state
  for (int i = 0; i < num_grid_elems; i++) {
    cur_state[i] = __float2half(-1.0);
  }
  cur_state[get_idx(GRID_SZ/2, GRID_SZ/2, 0)] = 0.0;

  for (int i = 0; i < GRID_SZ; i++) {
    for (int j = 0; j < GRID_SZ; j++) {
      cur_state[get_idx(i, j, GRID_SZ - 1)] = 1.0;
    }
  }

  while (true) {
    cudaMemset(phi, 1, grid_storage_bytes);
    //cudaMemcpy(phi, cur_state, grid_storage_bytes, cudaMemcpyDeviceToDevice);
    cudaCheckError();

    calculate_phi<<<numBlocks, threadsPerBlock>>>(cur_state, phi, scratch, rho);
    cudaCheckError();

    float *sum;
    cudaMallocManaged(&sum, sizeof(float));
    cudaCheckError();

    sum_arr<<<numBlocks, threadsPerBlock>>>(scratch, sum);
    cudaCheckError();

    cudaDeviceSynchronize();
    cudaCheckError(); // This catches any actual errors from calculate_phi
    std::cerr << "Error: " << *sum << std::endl;

    float *sel;
    cudaMallocManaged(&sel, sizeof(float));
    *sel = dis(gen);

    find_elem(phi, cur_state, sel);

    cudaDeviceSynchronize();
    cudaCheckError();

    //printf("%f\n", *sel);
    if (*sel == -1.0) {
      break;
    }
  }

  for (int i = 0; i < GRID_SZ; i++) {
    for (int j = 0; j < GRID_SZ; j++) {
      for (int k = 0; k < GRID_SZ; k++) {
        float val = __half2float(cur_state[get_idx(i, j, k)]);
        if (val == 0.0) {
          printf("%d %d %d\n", i, j, k);
        }
      }
    }
  }

  return 0;
}
