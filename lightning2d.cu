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

const unsigned GRID_SZ = 256;
const float h = 1.0/(GRID_SZ + 1.0);
const float h2 = h*h;

inline __host__ __device__ 
int get_idx(int x, int y)
{
  return y * GRID_SZ + x;
}

inline __host__ __device__
int get_idx(dim3 pos)
{
  return get_idx(pos.x, pos.y);
}

// Computes LHS of \del^2(\phi) = 4*\pi*\rho
__host__ __device__
void compute_error(float *phi_estimate, float *rho, float *err, dim3 pos) 
{
  float cur = phi_estimate[get_idx(pos)];

  float delx = 0;
  if (pos.x > 0) {
    delx += phi_estimate[get_idx(pos.x - 1, pos.y)] - cur;
  }

  if (pos.x < GRID_SZ - 1) {
    delx += phi_estimate[get_idx(pos.x + 1, pos.y)] - cur;
  }

  float dely = 0;
  if (pos.y > 0) {
    dely += phi_estimate[get_idx(pos.x, pos.y - 1)] - cur;
  }

  if (pos.y < GRID_SZ - 1) {
    dely += phi_estimate[get_idx(pos.x, pos.y + 1)] - cur;
  }

  err[get_idx(pos)] = abs(h2 * (delx + dely) + 4*M_PI * rho[get_idx(pos)]);
}

inline __device__ __host__
void relax_iter(float *state, float *phi, float *rho, dim3 pos)
{
  float cur_state = state[get_idx(pos)];
  if (cur_state != -1.0) {
    // Boundary condition
    phi[get_idx(pos)] = cur_state;
  } else {
    float cur_val = phi[get_idx(pos)];
    float cur_rho = -4.0*M_PI*rho[get_idx(pos)];

    float neighbor_sum = 0;
    int num_neighbors = 0;

    if (pos.x > 0) {
      neighbor_sum += phi[get_idx(pos.x - 1, pos.y)];
      num_neighbors++;
    }

    if (pos.x < GRID_SZ - 1) {
      neighbor_sum += phi[get_idx(pos.x + 1, pos.y)];
      num_neighbors++;
    }

    if (pos.y > 0) {
      neighbor_sum += phi[get_idx(pos.x, pos.y - 1)];
      num_neighbors++;
    }

    if (pos.y < GRID_SZ - 1) {
      neighbor_sum += phi[get_idx(pos.x, pos.y + 1)];
      num_neighbors++;
    }

    float new_val = (1.0 / (float)num_neighbors) * (neighbor_sum - h2 * cur_rho);

    // Overrelaxation
    float s = 1.7;
    new_val = s * new_val + (1.0 - s)*cur_val;

    phi[get_idx(pos)] = new_val;
  }
}

void calculate_phi(float *state, float *phi, float *tmp, float *rho)
{
  /* Initialization */
  for (int y = 0; y < GRID_SZ; y++) {
    for (int x = 0; x < GRID_SZ; x++) {
      dim3 pos(x, y);

      float cur_state = state[get_idx(pos)];
      if (cur_state == 0.0 || cur_state == 1.0) {
        phi[get_idx(pos)] = cur_state;
      } else {
        phi[get_idx(pos)] = 0.5;
      }
    }
  }

  /* Jacobi relaxation */
  const int phi_iters = 1000;
  for (int i = 0; i < phi_iters; i++) {
    for (int y = 0; y < GRID_SZ; y += 1) {
      for (int x = (y + 1) % 2; x < GRID_SZ; x += 2) {
        dim3 pos(x, y);
        relax_iter(state, phi, rho, pos);
      }
    }

    for (int y = 0; y < GRID_SZ; y += 1) {
      for (int x = y % 2; x < GRID_SZ; x += 2) {
        dim3 pos(x, y);
        relax_iter(state, phi, rho, pos);
      }
    }
  }

  for (int y = 0; y < GRID_SZ; y++) {
    for (int x = 0; x < GRID_SZ; x++) {
      dim3 pos(x, y);
      compute_error(phi, rho, tmp, pos);
    }
  }
}

__global__ 
void sum_arr(float *grid, float *sum)
{
  uint x = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint y = (blockIdx.y * blockDim.y) + threadIdx.y;

  float cur = grid[get_idx(x, y)];
  atomicAdd(sum, cur);
}

struct candidate {
  int x, y;
  float unnorm_prob;
};

bool neighbors_lightning(float *state, int x, int y) {
  if (x > 0 && state[get_idx(x - 1, y)] == 0.0) {
    return true;
  }

  if (x < GRID_SZ - 1 && state[get_idx(x + 1, y)] == 0.0) {
    return true;
  }

  if (y > 0 && state[get_idx(x, y - 1)] == 0.0) {
    return true;
  }

  if (y < GRID_SZ - 1 && state[get_idx(x, y + 1)] == 0.0) {
    return true;
  }

  return false;
}

bool find_elem(float *phi, float *state, float *rand) {
  float prob_sum = 0;
  std::vector<candidate> candidates;
  for (int y = 0; y < GRID_SZ; y++) {
    for (int x = 0; x < GRID_SZ; x++) {
      if (state[get_idx(x, y)] == 0.0) { // is lightning already
        continue;
      }
      if (neighbors_lightning(state, x, y)) {
        float val = phi[get_idx(x, y)];
        float unnorm_prob = val*val;
        candidates.emplace_back(candidate {x, y, unnorm_prob});
        prob_sum += unnorm_prob;
      }
    }
  }

  float cum = 0;
  for (auto &c : candidates) {
    cum += c.unnorm_prob / prob_sum;
    if (*rand < cum) {
      float *chosen = &state[get_idx(c.x, c.y)];
      if (*chosen == 1.0) {
        return true;
      } else {
        *chosen = 0;
        return false;
      }
    }
  }
  
  std::cout << *rand << " " << cum << std::endl;
  assert(false);

  return false;
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

  unsigned num_grid_elems = GRID_SZ*GRID_SZ;
  unsigned grid_storage_bytes = num_grid_elems*sizeof(float);

  float *state;
  float *phi;
  float *scratch;
  float *rho;

  cudaMallocManaged(&state, grid_storage_bytes);
  cudaCheckError();
  cudaMallocManaged(&phi, grid_storage_bytes);
  cudaCheckError();
  cudaMallocManaged(&scratch, grid_storage_bytes);
  cudaCheckError();
  cudaMallocManaged(&rho, grid_storage_bytes);
  cudaCheckError();

  cudaMemset(state, 0, grid_storage_bytes);
  cudaCheckError();
  cudaMemset(phi, 0, grid_storage_bytes);
  cudaCheckError();
  cudaMemset(scratch, 0, grid_storage_bytes);
  cudaCheckError();
  cudaMemset(rho, 0, grid_storage_bytes);
  cudaCheckError();

  dim3 threadsPerBlock(1, 1);
  dim3 numBlocks(GRID_SZ / threadsPerBlock.x, GRID_SZ / threadsPerBlock.y);

  // Initialize state
  for (int y = 0; y < GRID_SZ; y++) {
    for (int x = 0; x < GRID_SZ; x++) {
      state[get_idx(x, y)] = -1.0;
    }
  }

  state[get_idx(GRID_SZ/2, 0)] = 0.0;

  for (int x = 0; x < GRID_SZ; x++) {
    state[get_idx(x, GRID_SZ - 1)] = 1.0;
  }

  while (true) {
    //cudaMemcpy(phi, state, grid_storage_bytes, cudaMemcpyDeviceToDevice);
    cudaCheckError();

    //calculate_phi<<<numBlocks, threadsPerBlock>>>(state, phi, scratch, rho);
    calculate_phi(state, phi, scratch, rho);
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

    bool at_end = find_elem(phi, state, sel);

    cudaDeviceSynchronize();
    cudaCheckError();

    //printf("%f\n", *sel);
    if (at_end) {
      break;
    }
  }

  for (int i = 0; i < GRID_SZ; i++) {
    for (int j = 0; j < GRID_SZ; j++) {
      float val = state[get_idx(i, j)];
      if (val == 0.0) {
        printf("%d %d\n", i, j);
      }
    }
  }

  return 0;
}
