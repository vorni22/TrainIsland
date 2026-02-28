#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 3) in vec3 a_color;

out vec3 fragColor;

uniform mat4 Model;
uniform mat4 Projection;

void main() {
    gl_Position = Projection * Model * vec4(a_position, 1.0);
    fragColor = a_color;
}
