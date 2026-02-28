#version 330

// Input
in vec3 world_position;
in vec3 world_normal;
in vec3 object_color;

// Uniforms for light properties
uniform vec3 eye_position;
uniform float material_kd;
uniform float material_ks;
uniform int material_shininess;
uniform float alpha;
uniform int minimap;

struct light_source {
   int  type;
   vec3 position;
   vec3 color;
   vec3 direction;
};

uniform light_source lights[1];
uniform float cutoff_angle;

// Output
layout(location = 0) out vec4 out_color;

vec3 point_light_contribution(int id)
{
    float ambient_light = 0.1;
    vec3 light_color = lights[id].color;
    int type = lights[id].type;
    vec3 light_pos = lights[id].position;
    vec3 light_dir_custom = lights[id].direction;

    vec3 norm = normalize(world_normal);
    vec3 light_dir;

    if (type == 0) {
        // Point light
        light_dir = normalize(light_pos - world_position);
    } else if (type == 1) {
        // Spot light
        light_dir = normalize(light_pos - world_position);
    } else if (type == 2) {
        // Directional light
        light_dir = normalize(-light_dir_custom); // ensure it points *towards* the surface
    }

    vec3 view_dir = normalize(eye_position - world_position);

    vec3 ambient = ambient_light * light_color;
    vec3 diffuse = light_color * material_kd * max(dot(norm, light_dir), 0.0);
    vec3 specular = vec3(0);

    if (dot(norm, light_dir) > 0) {
        vec3 H = normalize(light_dir + view_dir);
        //specular = light_color * material_ks * pow(max(dot(norm, H), 0.0), material_shininess);
    }

    vec3 sum = diffuse + specular;

    // Spot light attenuation
    if (type == 1) {
        vec3 spot_light_dir = normalize(lights[id].direction);
        float cut_off = radians(cutoff_angle);
        float spot_dot = dot(-light_dir, spot_light_dir);
        float spot_limit = cos(cut_off);
        
        if (spot_dot > spot_limit) {
            float linear_att = (spot_dot - spot_limit) / (1.0 - spot_limit);
            float light_att_factor = pow(linear_att, 2);
            sum *= light_att_factor;
        } else {
            sum *= 0.0;
        }
    }

    return (sum + ambient) * object_color;
}

vec4 apply_fog(vec4 shadedColor) {
    if (minimap == 1)
        return shadedColor;

    float fog_maxdist = 600.0;
    float fog_mindist = 0.1;
    vec4  fog_colour = vec4(0.678, 0.941, 0.925, 1.0);

    // Calculate fog
    float dist = length(world_position - eye_position);
    float fog_factor = (fog_maxdist - dist) /
                    (fog_maxdist - fog_mindist);
    fog_factor = clamp(fog_factor, 0.0, 1.0);

    vec4 outputColor = mix(fog_colour, shadedColor, fog_factor);
    return outputColor;
}

void main()
{
    vec3 color = vec3(0);
    for (int i = 0; i < 1; i++) {
        color += point_light_contribution(i);
    }

    out_color = apply_fog(vec4(color, alpha));
}
