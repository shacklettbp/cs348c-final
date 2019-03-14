#include "camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdio>

void update_viewproj_mat(GLuint binding, int degrees, int grid_sz)
{
  glm::vec3 center(grid_sz / 2, grid_sz / 2, grid_sz / 2);

  float radius = (float)grid_sz / 2.f + 14.f;
  float rads = glm::radians((float)degrees);

  glm::vec3 camera_pos = center;

  camera_pos.x += cosf(rads) * radius;
  camera_pos.y += sinf(rads) * radius;
  camera_pos.z += 1;

  glm::mat4 model_mat = glm::mat4(1.f, 0.f, 0.f, 0.f,
                                  0.f, 1.f, 0.f, 0.f,
                                  0.f, 0.f, 1.f, 0.f,
                                  0.f, 0.f, 0.f, 1.f);

  glm::mat4 view_mat = glm::lookAt(camera_pos,
                                   center,
                                   glm::vec3(0, 0, -1));

  glm::mat4 proj_mat = glm::perspective(glm::radians(60.f), 1.0f, 0.1f, 1000.f);

  glm::mat4 vp = proj_mat * view_mat * model_mat;

  glUniformMatrix4fv(binding, 1, GL_FALSE, &vp[0][0]);
}
