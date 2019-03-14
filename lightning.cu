#include <iostream>
#include <fstream>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <math.h>
#include <cooperative_groups.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <random>
#include <cassert>
#include <vector>
#include <list>
#include <sstream>
#include <unordered_map>
#include <GLFW/glfw3.h>
#include <FreeImage.h>
#include "camera.h"
#include "apsf.h"

#ifndef NASSERT
#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                 \
  if (e != cudaSuccess) {                                              \
    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
    exit(0); \
  }                                                                 \
}
#else
#define cudaCheckError() {
}
#endif

static constexpr unsigned GRID_SZ = 64;
static constexpr int KERNEL_SZ = 513;
static constexpr int HALF_KERNEL_SZ = KERNEL_SZ / 2;
static constexpr unsigned IMG_SZ = 1080;
static const float h = 1.0/(GRID_SZ + 1.0);
static const float h2 = h*h;

std::random_device rd;  
std::mt19937 gen(rd()); 
std::uniform_real_distribution<> dis(0, 1.0);

static constexpr int THREADS_X = 16;
static constexpr int THREADS_Y = 16;
static constexpr int THREADS_Z = 4;
static const dim3 threadsPerBlock(THREADS_X, THREADS_Y, THREADS_Z);
static const dim3 numBlocks(GRID_SZ / threadsPerBlock.x, GRID_SZ / threadsPerBlock.y, GRID_SZ / threadsPerBlock.z);
static const dim3 halfNumBlocks(GRID_SZ / threadsPerBlock.x / 2, GRID_SZ / threadsPerBlock.y, GRID_SZ / threadsPerBlock.z);

static constexpr int numThreadsPerBlock = THREADS_X * THREADS_Y * THREADS_Z;
static constexpr int numWarps = numThreadsPerBlock / 32;
static_assert(numThreadsPerBlock == 1024);
static_assert(numWarps == 32);

static constexpr float exposure = 0.001;
static const float default_intensity = 25;
static const float secondary_intensity = 100;
static const float main_intensity = 1000;

struct GrowthSite
{
  dim3 pos;
  dim3 parent;
  bool term;
};

inline __device__
dim3 launchpos()
{
  uint x = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint y = (blockIdx.y * blockDim.y) + threadIdx.y;
  uint z = (blockIdx.z * blockDim.z) + threadIdx.z;
  
  return dim3(x, y, z);
}

inline __device__
unsigned get_block_thread()
{
  return threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
}

inline __host__ __device__ 
int get_idx(int x, int y, int z)
{
  return (z * GRID_SZ + y) * GRID_SZ + x;
}

inline __host__ __device__
int get_idx(dim3 pos)
{
  return get_idx(pos.x, pos.y, pos.z);
}

inline __device__
void relax_iter(float *state, float *phi, float *rho, dim3 pos)
{
  float cur_state = state[get_idx(pos)];
  if (cur_state != -1.0) {
    // Boundary condition
    phi[get_idx(pos)] = cur_state;
  } else {
    float cur_val = phi[get_idx(pos)];
    float cur_rho = rho[get_idx(pos)];

    float neighbor_sum = 0;
    int num_neighbors = 0;

    if (pos.x > 0) {
      neighbor_sum += phi[get_idx(pos.x - 1, pos.y, pos.z)];
      num_neighbors++;
    }

    if (pos.x < GRID_SZ - 1) {
      neighbor_sum += phi[get_idx(pos.x + 1, pos.y, pos.z)];
      num_neighbors++;
    }

    if (pos.y > 0) {
      neighbor_sum += phi[get_idx(pos.x, pos.y - 1, pos.z)];
      num_neighbors++;
    }

    if (pos.y < GRID_SZ - 1) {
      neighbor_sum += phi[get_idx(pos.x, pos.y + 1, pos.z)];
      num_neighbors++;
    }

    if (pos.z > 0) {
      neighbor_sum += phi[get_idx(pos.x, pos.y, pos.z - 1)];
      num_neighbors++;
    }

    if (pos.z < GRID_SZ - 1) {
      neighbor_sum += phi[get_idx(pos.x, pos.y, pos.z + 1)];
      num_neighbors++;
    }

    float new_val = (1.0 / (float)num_neighbors) * (neighbor_sum - h2 * cur_rho);

    // Overrelaxation
    float s = 1.8;
    new_val = s * new_val + (1.0 - s)*cur_val;

    phi[get_idx(pos)] = new_val;
  }
}

__device__ __host__
bool neighbors_lightning(float *state, dim3 pos) {
  if (pos.x > 0 && state[get_idx(pos.x - 1, pos.y, pos.z)] == 0.0) {
    return true;
  }

  if (pos.x < GRID_SZ - 1 && state[get_idx(pos.x + 1, pos.y, pos.z)] == 0.0) {
    return true;
  }

  if (pos.y > 0 && state[get_idx(pos.x, pos.y - 1, pos.z)] == 0.0) {
    return true;
  }

  if (pos.y < GRID_SZ - 1 && state[get_idx(pos.x, pos.y + 1, pos.z)] == 0.0) {
    return true;
  }

  if (pos.z > 0 && state[get_idx(pos.x, pos.y, pos.z - 1)] == 0.0) {
    return true;
  }

  if (pos.z < GRID_SZ - 1 && state[get_idx(pos.x, pos.y, pos.z + 1)] == 0.0) {
    return true;
  }

  return false;
}

__global__
void phi_init(float *state, float *phi)
{
  dim3 pos = launchpos();

  float cur_state = state[get_idx(pos)];
  if (cur_state == 0.0 || cur_state == 1.0) {
    phi[get_idx(pos)] = cur_state;
  } else {
    phi[get_idx(pos)] = 0.5;
  }
}

__global__
void calculate_phi_redblack(float *state, float *phi, float *rho, int red)
{
  dim3 launched = launchpos();
  dim3 pos = dim3(launched.x * 2 + ((launched.z + launched.y + red) % 2), launched.y, launched.z);
  relax_iter(state, phi, rho, pos);
}

__device__
float warp_sum(float sum) 
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }

  return sum;
}

__device__
float warp_scan(float sum, int lane)
{
  for (int i = 1; i < warpSize; i *= 2) {
    float prev = __shfl_up_sync(0xFFFFFFFF, sum, i, warpSize);

    if (lane >= i) {
      sum += prev;
    }
  }

  return sum;
}

__device__
void reduce_error(float err, float *out)
{
  static __shared__ float shared[numWarps];

  unsigned thread_id = get_block_thread();
  unsigned thread_lane = thread_id % warpSize;
  unsigned warp_id = thread_id / warpSize;

  float sum = warp_sum(err);

  // Master in warp writes
  if (thread_lane == 0) {
    shared[warp_id] = sum;
  }

  __syncthreads();

  // First warp now reads shared memory
  sum = shared[thread_lane];
  float block_sum = warp_sum(sum);

  // Master thread atomically writes
  if (thread_id == 0) {
    atomicAdd(out, block_sum);
  }
}

__global__
void calculate_phi_error(float *state, float *phi, float *err_out, float *rho)
{
  dim3 pos = launchpos();
  float cur = phi[get_idx(pos)];

  float delx = 0;
  if (pos.x > 0) {
    delx += phi[get_idx(pos.x - 1, pos.y, pos.z)] - cur;
  }

  if (pos.x < GRID_SZ - 1) {
    delx += phi[get_idx(pos.x + 1, pos.y, pos.z)] - cur;
  }

  float dely = 0;
  if (pos.y > 0) {
    dely += phi[get_idx(pos.x, pos.y - 1, pos.z)] - cur;
  }

  if (pos.y < GRID_SZ - 1) {
    dely += phi[get_idx(pos.x, pos.y + 1, pos.z)] - cur;
  }

  float delz = 0;
  if (pos.z > 0) {
    delz += phi[get_idx(pos.x, pos.y, pos.z - 1)] - cur;
  }

  if (pos.z < GRID_SZ - 1) {
    delz += phi[get_idx(pos.x, pos.y, pos.z + 1)] - cur;
  }

  float err = abs(h2 * (delx + dely + delz) - rho[get_idx(pos)]);

  reduce_error(err, err_out);
}

void calculate_phi(float *state, float *phi, float *rho)
{
  phi_init<<<numBlocks, threadsPerBlock>>>(state, phi);
  cudaCheckError();

  const int phi_iters = 100;
  for (int iter = 0; iter < phi_iters; iter++) {
    calculate_phi_redblack<<<halfNumBlocks, threadsPerBlock>>>(state, phi, rho, 1);
    cudaCheckError();
    calculate_phi_redblack<<<halfNumBlocks, threadsPerBlock>>>(state, phi, rho, 0);
    cudaCheckError();
  }
}

__device__
void weight_reduce(float *out, float *total_sum_out, float weight, dim3 pos)
{
  static __shared__ float shared[numWarps];

  unsigned thread_id = get_block_thread();
  unsigned thread_lane = thread_id % warpSize;
  unsigned warp_id = thread_id / warpSize;

  float sum = warp_scan(weight, thread_lane);

  // Master in warp writes to shared block memory
  if (thread_lane == warpSize - 1) {
    shared[warp_id] = sum;
  }

  __syncthreads();

  // Master warp
  if (warp_id == 0) {
    float warp_sum = shared[thread_lane];
    float block_sum = warp_scan(warp_sum, thread_lane);

    shared[thread_lane] = block_sum;
  }
  __syncthreads();

  if (warp_id > 0) {
    sum += shared[warp_id - 1];
  }

  if (thread_id == numThreadsPerBlock - 1) {
    float running_sum = atomicAdd(total_sum_out, sum);
    shared[0] = running_sum;
  }

  __syncthreads();

  sum += shared[0];

  out[get_idx(pos)] = sum;
}

__global__
void compute_growth_weights(float *phi, float *state, float *scratch, float *total_sum)
{
  dim3 pos = launchpos();

  float weight;
  if (state[get_idx(pos)] == 0.0 /* lightning already */ || !neighbors_lightning(state, pos)) {
    weight = 0;
  } else {
    float val = phi[get_idx(pos)];
    // Assumes eta = 2
    weight = val*val;
  }

  // Save so find next growth doesn't need to recompute
  phi[get_idx(pos)] = weight;
  
  weight_reduce(scratch, total_sum, weight, pos);
}

__global__
void find_next_growth(float *phi, float *state, float *weights, float *weight_total, GrowthSite *growth, float rand)
{
  dim3 pos = launchpos();

  float weight = phi[get_idx(pos)];
  float search = *weight_total * rand;


  float cumulative_sum = weights[get_idx(pos)];

  if (search >= cumulative_sum - weight && search < cumulative_sum) {
    if (pos.x > 0 && state[get_idx(pos.x - 1, pos.y, pos.z)] == 0.0) {
      growth->parent = dim3(pos.x - 1, pos.y, pos.z);
    }
    else if (pos.x < GRID_SZ - 1 && state[get_idx(pos.x + 1, pos.y, pos.z)] == 0.0) {
      growth->parent = dim3(pos.x + 1, pos.y, pos.z);
    }
    else if (pos.y > 0 && state[get_idx(pos.x, pos.y - 1, pos.z)] == 0.0) {
      growth->parent = dim3(pos.x, pos.y - 1, pos.z);
    }
    else if (pos.y < GRID_SZ - 1 && state[get_idx(pos.x, pos.y + 1, pos.z)] == 0.0) {
      growth->parent = dim3(pos.x, pos.y + 1, pos.z);
    }
    else if (pos.z > 0 && state[get_idx(pos.x, pos.y, pos.z - 1)] == 0.0) {
      growth->parent = dim3(pos.x, pos.y, pos.z - 1);
    }
    else if (pos.z < GRID_SZ - 1 && state[get_idx(pos.x, pos.y, pos.z + 1)] == 0.0) {
      growth->parent = dim3(pos.x, pos.y, pos.z + 1);
    } 
    else {
      return;
    }

    float cur_state = state[get_idx(pos)];
    if (cur_state == 1.0) {
      growth->term = true;
    } else {
      state[get_idx(pos)] = 0.0;
      growth->term = false;
    }
    growth->pos = pos;
  }
}

struct LightningNode
{
  LightningNode *parent;
  std::vector<LightningNode *> children;

  float x, y, z;
  dim3 grid_pos;
  float intensity;
  int index;
};

void write_secondary_intensities(LightningNode *cur)
{
  int num_children = cur->children.size();
  if (num_children == 0) {
    return;
  }

  int idx = (int)(dis(gen) * (float)num_children);

  LightningNode *child = cur->children[idx];
  if (child->intensity == default_intensity) { 
    child->intensity = secondary_intensity;
  }
  write_secondary_intensities(child);
}

static GLuint vao, vbo[2], ebo, render_texture, fbo, mvp;
struct cudaGraphicsResource *cuda_render_tex_resource;

texture<float4, cudaTextureType2D, cudaReadModeElementType> cuda_tex_ref;
__global__
void apsf_and_tonemap(float3 *out, float *apsf)
{
  dim3 pos = launchpos();
  if (pos.y >= IMG_SZ || pos.x >= IMG_SZ) {
    return;
  }

  float result_x = 0;
  float result_y = 0;
  float result_z = 0;

  for (int y = -HALF_KERNEL_SZ; y <= HALF_KERNEL_SZ; y++) {
    for (int x = -HALF_KERNEL_SZ; x <= HALF_KERNEL_SZ; x++) {
      int kernel_idx = (y + HALF_KERNEL_SZ) * KERNEL_SZ + (x + HALF_KERNEL_SZ);

      int tex_x = pos.x + x;
      int tex_y = pos.y + y;

      if (tex_x < 0 || tex_x >= IMG_SZ || tex_y < 0 || tex_y >= IMG_SZ) {
        continue;
      }

      float apsf_val = apsf[kernel_idx];

      float4 v = tex2D(cuda_tex_ref, tex_y, tex_x);

      result_x += apsf_val * v.x;
      result_y += apsf_val * v.y;
      result_z += apsf_val * v.z;
    }
  }

  //float4 v = tex2D(cuda_tex_ref, pos.y, pos.x);

  //result_x = v.x;
  //result_y = v.y;
  //result_z = v.z;

  result_x *= exposure;
  result_y *= exposure;
  result_z *= exposure;

  unsigned idx = IMG_SZ * pos.y + pos.x;
  out[idx].x = result_x / (result_x + 1);
  out[idx].y = result_y / (result_y + 1);
  out[idx].z = result_z / (result_z + 1);
}

static int saved_imgs = 0;

void save_img(float3 *result) {
  FIBITMAP *bitmap = FreeImage_Allocate(IMG_SZ, IMG_SZ, 24);

  for (int y = 0; y < IMG_SZ; y++) {
    for (int x = 0; x < IMG_SZ; x++) {
      float3 cur = result[y * IMG_SZ + x];
      RGBQUAD color;
      color.rgbRed = cur.x * 255;
      color.rgbGreen  = cur.y * 255;
      color.rgbBlue  = cur.z * 255;

      FreeImage_SetPixelColor(bitmap, y, x, &color);
    }
  }

  FreeImage_Save(FIF_PNG, bitmap, (std::string("renders/") + std::to_string(saved_imgs++) + std::string(".png")).c_str(), 0);
  FreeImage_Unload(bitmap);
}

void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message) {
  printf("\n*** ");
  if (fif != FIF_UNKNOWN) {
    printf("%s Format\n", FreeImage_GetFormatFromFIF(fif));
  }
  printf(message);
  printf(" ***\n");
}

void render_lightning(const std::list<LightningNode> &nodes, float *apsf) 
{
  std::vector<float> vertices;
  std::vector<float> intensities;
  std::vector<unsigned int> indices;
  for (const LightningNode &node : nodes) {
    vertices.push_back(node.x);
    vertices.push_back(node.y);
    vertices.push_back(node.z);

    intensities.push_back(node.intensity);

    if (node.parent == nullptr) {
      continue;
    }

    indices.push_back(node.index);
    indices.push_back(node.parent->index);
  }

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);
  int vertex_attr = 0;
  glEnableVertexAttribArray(vertex_attr);
  glVertexAttribPointer(vertex_attr, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, intensities.size() * sizeof(float), &intensities[0], GL_STATIC_DRAW);
  int intensity_attr = 1;
  glEnableVertexAttribArray(intensity_attr);
  glVertexAttribPointer(intensity_attr, 1 /* size  */, GL_FLOAT, GL_FALSE, 0, nullptr);

  glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_INT, nullptr);

  cudaGraphicsMapResources(1, &cuda_render_tex_resource, 0);
  cudaCheckError();

  cudaArray *arr;
  cudaGraphicsSubResourceGetMappedArray(&arr, cuda_render_tex_resource, 0, 0);
  cudaCheckError();

  cudaBindTextureToArray(cuda_tex_ref, arr);
  cudaCheckError();

  float3 *render_result;
  cudaMallocManaged(&render_result, sizeof(float)*3*IMG_SZ*IMG_SZ);

  dim3 num_img_threads(32, 32);
  dim3 num_img_blocks(IMG_SZ / num_img_threads.x + 1, IMG_SZ / num_img_threads.y + 1);
  apsf_and_tonemap<<<num_img_blocks, num_img_threads>>>(render_result, apsf);
  cudaCheckError();

  cudaDeviceSynchronize();
  cudaCheckError();

  save_img(render_result);

  cudaFree(render_result);
  cudaGraphicsUnmapResources(1, &cuda_render_tex_resource, 0);
  cudaCheckError();

  glClear(GL_COLOR_BUFFER_BIT);
}

static int deg = 0;

void draw_lightning(const std::vector<GrowthSite *> &bolt, float *apsf, float *rho)
{
  std::list<LightningNode> node_list;
  std::unordered_map<unsigned, LightningNode *> node_map;

  dim3 root_pos(GRID_SZ / 2, GRID_SZ / 2, 0);
  int index = 0;
  node_list.emplace_back(LightningNode { nullptr, {}, (float)root_pos.x, (float)root_pos.y, (float)root_pos.z, root_pos, main_intensity, index++});
  LightningNode *root = &node_list.back();
  node_map[get_idx(GRID_SZ / 2, GRID_SZ / 2, 0)] = root;

  LightningNode *terminator = nullptr;

  for (const GrowthSite *growth : bolt) {
    unsigned idx = get_idx(growth->pos);
    unsigned parent_idx = get_idx(growth->parent);

    auto parent_iter = node_map.find(parent_idx);
    if (parent_iter == node_map.end()) {
      return;
    }

    LightningNode *parent = parent_iter->second;

    node_list.emplace_back(
        LightningNode { parent, {},
                        (float)growth->pos.x + (float)dis(gen) - 0.5f,
                        (float)growth->pos.y + (float)dis(gen) - 0.5f,
                        (float)growth->pos.z + (float)dis(gen) - 0.5f,
                        growth->pos,
                        default_intensity, index++});

    LightningNode *cur = &node_list.back();
    parent->children.push_back(cur);

    node_map[idx] = cur;

    if (growth->term) {
      terminator = cur;
    }
  }

  // Set channel values
  LightningNode *cur = terminator;
  while (cur != nullptr) {
    write_secondary_intensities(cur);

    cur->intensity = main_intensity;

    rho[get_idx(cur->grid_pos)] = 100;

    cur = cur->parent;
  }

  for (int i = 0; i < 1; i++) {
    update_viewproj_mat(mvp, deg, GRID_SZ);

    deg = (deg + 1) % 360;
    render_lightning(node_list, apsf);
  }
}

void simulate_and_draw_bolt(float *state, float *phi, float *scratch, float *rho, float *apsf)
{
  float *weight_total;
  cudaMallocManaged(&weight_total, sizeof(float));

  std::vector<GrowthSite *> bolt;
  while (true) {
    calculate_phi(state, phi, rho);

    GrowthSite *growth;
    cudaMallocManaged(&growth, sizeof(GrowthSite));
    growth->pos.x = GRID_SZ + 1;
    growth->pos.y = GRID_SZ + 1;
    growth->pos.z = GRID_SZ + 1;

    //scratch[0] = 0;
    //calculate_phi_error<<<numBlocks, threadsPerBlock>>>(state, phi, scratch, rho);
    //cudaCheckError();

    //cudaDeviceSynchronize();
    //cudaCheckError(); // This catches any actual errors from cuda
    //std::cerr << "Error: " << scratch[0] << std::endl;

    *weight_total = 0;
    compute_growth_weights<<<numBlocks, threadsPerBlock>>>(phi, state, scratch, weight_total);
    cudaCheckError();

    float rand = dis(gen);
    find_next_growth<<<numBlocks, threadsPerBlock>>>(phi, state, scratch, weight_total, growth, rand);
    cudaCheckError();

    cudaDeviceSynchronize();
    cudaCheckError();

    if (growth->pos.x == GRID_SZ + 1) {
      continue;
    }

    bolt.push_back(growth);
    if (growth->term) {
      break;
    }
  }

  cudaMemset(rho, 0, GRID_SZ*GRID_SZ*GRID_SZ*sizeof(float));

  draw_lightning(bolt, apsf, rho);

  cudaFree(weight_total);
  for (GrowthSite *segment : bolt) {
    cudaFree(segment);
  }
}

// Initialize state. Lightning starts at top,
// ground at bottom.
void init_state(float *state)
{
  for (int z = 0; z < GRID_SZ; z++) {
    for (int y = 0; y < GRID_SZ; y++) {
      for (int x = 0; x < GRID_SZ; x++) {
        state[get_idx(x, y, z)] = -1.0;
      }
    }
  }

  state[get_idx(GRID_SZ/2, GRID_SZ/2, 0)] = 0.0;

  for (int x = 0; x < GRID_SZ; x++) {
    for (int y = 0; y < GRID_SZ; y++) {
      state[get_idx(x, y, GRID_SZ - 1)] = 1.0;
    }
  }
}

void compile_shader(GLuint shader, const std::string &filename)
{
  std::ifstream file(filename);

  std::string contents((std::istreambuf_iterator<char>(file)),
                       (std::istreambuf_iterator<char>()));

  const char *raw = contents.c_str();

  glShaderSource(shader, 1, &raw, nullptr);
  glCompileShader(shader);

  GLint res = GL_FALSE;
  int log_size;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &res);
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_size);
  if (res == GL_FALSE || log_size > 0) {
    std::vector<char> err_msg(log_size+1);
    glGetShaderInfoLog(shader, log_size, nullptr, err_msg.data());

    std::cerr << err_msg.data() << std::endl;
    exit(1);
  }

  file.close();
}

GLFWwindow * setup_renderer()
{
  int res = glfwInit();
  assert(res);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_VISIBLE, false);

  GLFWwindow *window = glfwCreateWindow(IMG_SZ, IMG_SZ, "LIGHTNING", nullptr, nullptr);
  glfwMakeContextCurrent(window);

  res = glewInit();
  assert(res == GLEW_OK);

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(2, vbo);
  glGenBuffers(1, &ebo);

  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE);
  
  glGenTextures(1, &render_texture);
  glBindTexture(GL_TEXTURE_2D, render_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, IMG_SZ, IMG_SZ, 0, GL_RGBA, GL_FLOAT, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, render_texture, 0);
  GLenum draw_bufs[1] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, draw_bufs);
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

  glViewport(0,0,IMG_SZ,IMG_SZ);

  GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
  GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

  compile_shader(vert_shader, "vert.glsl");
  compile_shader(frag_shader, "frag.glsl");

  GLuint prog = glCreateProgram();
  glAttachShader(prog, vert_shader);
  glAttachShader(prog, frag_shader);
  glLinkProgram(prog);

  int log_size;
  glGetProgramiv(prog, GL_LINK_STATUS, &res);
  glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &log_size);
  if (res == GL_FALSE || log_size > 0) {
    std::vector<char> err_msg(log_size+1);
    glGetProgramInfoLog(prog, log_size, nullptr, err_msg.data());

    std::cerr << "Linkage failed\n" ;
    std::cerr << err_msg.data() << std::endl;
    exit(1);
  }

  glDetachShader(prog, vert_shader);
  glDetachShader(prog, frag_shader);

  glDeleteShader(vert_shader);
  glDeleteShader(frag_shader);

  glUseProgram(prog);

  glLineWidth(2);

  mvp = glGetUniformLocation(prog, "MVP");

  FreeImage_Initialise();
  FreeImage_SetOutputMessage(FreeImageErrorHandler);

  cudaGraphicsGLRegisterImage(&cuda_render_tex_resource, render_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
  cudaCheckError();

  return window;
}

int main(void)
{
  cudaSetDevice(0);
  size_t free_bytes, total_bytes;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  //printf("Free bytes on device: %lu\n", free_bytes);

  unsigned num_grid_elems = GRID_SZ*GRID_SZ*GRID_SZ;
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

  float *apsf = make_apsf(KERNEL_SZ);

  auto window = setup_renderer();

  while (true) {
    init_state(state);

    simulate_and_draw_bolt(state, phi, scratch, rho, apsf);
  }

  glfwDestroyWindow(window);

  FreeImage_DeInitialise();
  return 0;
}
