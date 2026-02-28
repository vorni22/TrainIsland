#version 330 core

in vec3 fragColor;

uniform int isHighlighted;
uniform vec3 highlightColor;

out vec4 outColor;

void main()
{
    vec3 col = fragColor;

    if (isHighlighted == 1)
        col = highlightColor;

    outColor = vec4(col, 1.0);
}
