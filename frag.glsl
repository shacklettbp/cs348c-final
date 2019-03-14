#version 450 core

in float intensity;

layout(location = 0) out vec4 color;

void main() {
  vec4 lightning_color = vec4(0.22, 0.43, 1, 1);

  color = intensity * lightning_color;
}
