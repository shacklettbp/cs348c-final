#version 450 core

layout(location = 0) in vec3 vert_pos_world;
layout(location = 1) in float vert_intensity;

out float intensity;

uniform mat4 MVP;

void main() {
  gl_Position = MVP * vec4(vert_pos_world, 1);

  intensity = vert_intensity;
}
