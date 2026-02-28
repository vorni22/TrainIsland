#include "Terrain.h"
#include <iostream>

float random_float(float a, float b) {
    return a + (b - a) * ((float)rand() / (float)RAND_MAX);
}

int random_int(int a, int b) {
    return a + rand() % (b - a + 1);
}

glm::vec3 find_barycentric2d(const glm::vec2 &A, const glm::vec2 &B, const glm::vec2 &C, const glm::vec2 &P)
{
    glm::vec2 v0 = B - A, v1 = C - A, v2 = P - A;
    float d00 = glm::dot(v0, v0);
    float d01 = glm::dot(v0, v1);
    float d11 = glm::dot(v1, v1);
    float d20 = glm::dot(v2, v0);
    float d21 = glm::dot(v2, v1);

    float det = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / det;
    float w = (d00 * d21 - d01 * d20) / det;
    float u = 1.0f - v - w;

    return glm::vec3(u, v, w);
}

bool is_inside_triangle(const glm::vec2 &A, const glm::vec2 &B, const glm::vec2 &C, const glm::vec2 &P) {
    glm::vec3 uvw = find_barycentric2d(A, B, C, P);
    return
        uvw.x >= 0.0f && uvw.y >= 0.0f && uvw.z >= 0.0f &&
       (uvw.x + uvw.y + uvw.z <= 1.0001f);
}

bool is_inside_quad(const glm::vec2 &A, const glm::vec2 &B, const glm::vec2 &C, const glm::vec2 &D, const glm::vec2 &P) {
    return is_inside_triangle(A, B, C, P) || is_inside_triangle(A, C, D, P);
}

float distance_to_line2d(glm::vec2 P0, glm::vec2 P1, glm::vec2 P, float &t) {
    glm::vec2 P0P1 = P1 - P0;
    float d = glm::dot(P0P1, P0P1);
    if (std::abs(d) < 1e-7f) {
        t = std::numeric_limits<float>::infinity();
        return t;
    }
    t = glm::dot(P0P1, P - P0) / d;
    return glm::length(P - P0 - t * P0P1);
}


float compute_abrupt_falloff(float t) {
    t = glm::clamp(t, 0.0f, 1.0f);
    return std::pow((1.0f - t), 4.0f);
}

glm::vec3 plane_normal(glm::vec3 A, glm::vec3 B, glm::vec3 C) {
    glm::vec3 norm = -glm::normalize(glm::cross(B - A, C - A));
    //if (norm.y < 0.01f)
    //    norm = -norm;

    return norm;
}

// =====================================================================================================================================

Terrain::Terrain(int lenx, int leny, create_mesh_func create_mesh) {
    lx = lenx;
    ly = leny;

    game_over = false;
    mission_time = 0.0f;
    internal_time = 0.0f;

    heightmap.resize(lenx + 1, std::vector<float>(leny + 1, 0.0f));
    normalmap.resize(lenx + 1, std::vector<glm::vec3>(leny + 1, glm::vec3(0.0f)));
    filter_map.resize(lenx + 1, std::vector<bool>(leny + 1, true));

    rails.resize(lenx / GRID_CELL_SIZE, std::vector<float>(leny / GRID_CELL_SIZE, -1.0f));
    rail_heightmap.resize(lenx / GRID_CELL_SIZE, std::vector<float>(leny / GRID_CELL_SIZE, -1.0f));

    perlin::Perlin2D noise(2.0f);
    river_noise = new perlin::Perlin2D(0.0f);
    
    float cx = lx * 0.5f;
    float cy = ly * 0.5f;
    float sharpness = 3.0f;
    float edge_variation = 0.25f;
    float amplitude = 30.0f;
    float scale = 0.01f;

    generate_rivers(2);

    max_h = 0.0f;
    min_h = amplitude;

    for (int x = 0; x <= lx; x++) {
        for (int y = 0; y <= ly; y++) {
            float river_factor = get_river_noise(x, y);
            //heightmap[x][y] = river_factor * amplitude;
            //continue;

            float h = noise.perlin_fractal(x * scale, y * scale, 6) * (amplitude + 5.0f);

            // Distance from center
            float dx = (x - cx) / cx;
            float dy = (y - cy) / cy;
            float dist = sqrtf(dx*dx + dy*dy);

            float noise_factor = noise.perlin_fractal(x * 0.005f, y * 0.005f, 3);
            dist *= (1.0f + (noise_factor - 0.5f) * edge_variation);

            float falloff = glm::clamp(1.0f - pow(dist, sharpness), 0.0f, 1.0f);
            h *= falloff;

            heightmap[x][y] = h * river_factor;

            max_h = std::max(max_h, heightmap[x][y]);
            min_h = std::min(min_h, heightmap[x][y]);
        }
    }

    generate_rails(8);
    create_grid();

    for (int i = 0; i < 4; i++) {
        std::vector<VertexFormat> vertices;
        std::vector<uint32_t> indices;
        Resources::create_resourve(i, vertices, indices);
        create_mesh(("resource_" + std::to_string(i)).c_str(), vertices, indices);
    }

    {
        std::vector<VertexFormat> vertices;
        std::vector<uint32_t> indices;
        Resources::draw_cube(glm::vec3(0.0, 0.0f, 0.0f), 0.5f, glm::vec3(0,0,1), vertices, indices);
        create_mesh("ping", vertices, indices);
    }

    generate_request(5);

    // Loop over each quad and compute triangle normals
    for (int x = 0; x < lx; x++) {
        for (int y = 0; y < ly; y++) {
            glm::vec3 A(x, heightmap[x][y], y);
            glm::vec3 B(x + 1, heightmap[x + 1][y], y);
            glm::vec3 C(x + 1, heightmap[x + 1][y + 1], y + 1);
            glm::vec3 D(x, heightmap[x][y + 1], y + 1);

            glm::vec3 n1 = glm::normalize(glm::cross(B - A, C - A));
            if (n1.y < 0)
                n1 = -n1;

            glm::vec3 n2 = glm::normalize(glm::cross(C - A, D - A));
            if (n2.y < 0)
                n2 = -n2;

            normalmap[x][y] += n1 + n2;         // A
            normalmap[x + 1][y] += n1;          // B
            normalmap[x + 1][y + 1] += n1 + n2; // C
            normalmap[x][y + 1] += n2;          // D
        }
    }

    // Normalize all normals
    for (int x = 0; x <= lx; x++) {
        for (int y = 0; y <= ly; y++) {
            normalmap[x][y] = glm::normalize(normalmap[x][y]);
        }
    }

    for (int x = 0; x < lx; x++) {
        for (int y = 0; y < ly; y++) {
            float grid_x = x / GRID_CELL_SIZE;
            float grid_y = y / GRID_CELL_SIZE;

            float min_val = heightmap[x][y];
            float max_val = heightmap[x][y];

            min_val = std::min(
                min_val,
                std::min(
                    heightmap[x][y + 1],
                    std::min(
                        heightmap[x][y + 1],
                        heightmap[x + 1][y + 1]
                    )
                )
            );

            max_val = std::min(
                max_val,
                std::min(
                    heightmap[x][y + 1],
                    std::min(
                        heightmap[x][y + 1],
                        heightmap[x + 1][y + 1]
                    )
                )
            );

            float value_here = rails[grid_x][grid_y];

            if (value_here >= 0.0f && min_val >= value_here - 1.0f && min_val - value_here <= 4.0f)
                filter_map[x][y] = false;
        }
    }
}

void Terrain::generate_mesh(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    vertices.resize((lx + 1) * (ly + 1), VertexFormat(glm::vec3(0)));
    indices.clear();

    auto vert_id = [this](int x, int y) {
        return x * (ly + 1) + y;
    };

    //min_h = 0.0f;
    //max_h = 30.0f;

    for (int x = 0; x <= lx; x++) {
        for (int y = 0; y <= ly; y++) {
            glm::vec3 pos = glm::vec3(x, heightmap[x][y], y);
            glm::vec3 normal = normalmap[x][y];

            glm::vec3 color;

            // Calculate normalized height
            float h = heightmap[x][y];
            float hn = (heightmap[x][y] - min_h) / (max_h - min_h);

            float valley_flat = 0.3f;
            float hn_remap;

            if (hn < 0.4f) {
                hn_remap = hn * valley_flat / 0.4f;
            } else {
                hn_remap = valley_flat + (1.0f - valley_flat) * (hn - 0.4f) / 0.6f;
            }

            glm::vec3 grass_color = glm::vec3(0.0f,0.5f,0.0f);
            glm::vec3 dirt_color  = glm::vec3(0.5f,0.35f,0.05f);
            glm::vec3 rock_color  = glm::vec3(0.6f,0.6f,0.6f);
            glm::vec3 snow_color  = glm::vec3(1.0f);

            float beach_width = 1.5f;
            float base_h = h;
            float base_color_factor = glm::clamp((base_h - WATER_LEVEL)/beach_width, 0.0f, 1.0f);
            glm::vec3 base_color = glm::mix(glm::vec3(0.94f,0.87f,0.65f), grass_color, base_color_factor);

            float t1 = glm::clamp(glm::smoothstep(0.0f,0.5f,hn_remap),0.0f,1.0f);
            float t2 = glm::clamp(glm::smoothstep(0.4f,0.8f,hn_remap),0.0f,1.0f);
            float t3 = glm::clamp(glm::smoothstep(0.7f,1.0f,hn_remap),0.0f,1.0f);

            glm::vec3 terrain_color = glm::mix(base_color, glm::mix(glm::mix(grass_color,dirt_color,t2), glm::mix(rock_color,snow_color,t3), t3), t1);

            color = terrain_color;

            vertices[vert_id(x, y)] = VertexFormat(pos, color, normal);
        }
    }

    for (int x = 0; x < lx; x++) {
        for (int y = 0; y < ly; y++) {
            if (!filter_map[x][y])
                continue;

            unsigned int A = vert_id(x, y);
            unsigned int B = vert_id(x + 1, y);
            unsigned int C = vert_id(x + 1, y + 1);
            unsigned int D = vert_id(x, y + 1);

            indices.push_back(B);
            indices.push_back(A);
            indices.push_back(C);

            indices.push_back(D);
            indices.push_back(C);
            indices.push_back(A);
        }
    }
}

void Terrain::generate_rail_mesh(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    // draw rails
    for (int x = 0; x < lx / GRID_CELL_SIZE; x++) {
        for (int y = 0; y < ly / GRID_CELL_SIZE; y++) {
            draw_rail_cell(x, y, vertices, indices, true);
        }
    }

    for (int i = 0; i < station_pos.size(); i++) {
        draw_station(i, vertices, indices);
    }
}

void Terrain::set_grid_cell(int x, int y, float value) {
    if (x < 0 || y < 0 || x + 1 >= lx / GRID_CELL_SIZE || y + 1 >= ly / GRID_CELL_SIZE)
        return;

    for (int sx = x * GRID_CELL_SIZE; sx < x * GRID_CELL_SIZE + GRID_CELL_SIZE; sx++) {
        for (int sy = y * GRID_CELL_SIZE; sy < y * GRID_CELL_SIZE + GRID_CELL_SIZE; sy++) {
            heightmap[sx][sy] = value;
            heightmap[sx + 1][sy] = value;
            heightmap[sx][sy + 1] = value;
            heightmap[sx + 1][sy + 1] = value;
        }
    }

    rail_heightmap[x][y] = value;
}

float Terrain::get_grid_cell(int x, int y) {
    float value = 0.0f;

    if (rail_heightmap[x][y] >= 0.0f)
        return rail_heightmap[x][y];

    int cnt = 0;
    for (int sx = x * GRID_CELL_SIZE; sx < x * GRID_CELL_SIZE + GRID_CELL_SIZE; sx++) {
        for (int sy = y * GRID_CELL_SIZE; sy < y * GRID_CELL_SIZE + GRID_CELL_SIZE; sy++) {
            value += heightmap[sx][sy] + heightmap[sx + 1][sy] + heightmap[sx][sy + 1] + heightmap[sx + 1][sy + 1];
            cnt += 4;
        }
    }

    return value / cnt;
}

void Terrain::generate_rivers(int cnt) {
    rivers.clear();

    srand(time(NULL));
    
    int current_side = 0;

    float offsert = 10.0f;
    for (int i = 0; i < cnt; i++) {
        int side_start = current_side;
        int side_end = (side_start + 2) % 4;

        current_side = (current_side + 1) % 4;

        float source_x = random_float(0.2 * lx, 0.8 * lx);
        float source_y = random_float(0.2 * ly, 0.8 * ly);

        float end_x = random_float(0.2 * lx, 0.8 * lx);
        float end_y = random_float(0.2 * ly, 0.8 * ly);

        switch (side_start)
        {
        case 0:
            source_y = 0.0f;
            break;
        case 1:
            source_x = lx * 1.0f;
            break;
        case 2:
            source_y = ly * 1.0f;
            break;
        case 3:
            source_x = 0.0f;
            break;
        }

        switch (side_end)
        {
        case 0:
            end_y = 0.0f;
            break;
        case 1:
            end_x = lx * 1.0f;
            break;
        case 2:
            end_y = ly * 1.0f;
            break;
        case 3:
            end_x = 0.0f;
            break;
        }

        float inv_size_factor = 0.25f;

        float width = random_float(0.06f, 0.12f) * 3.0f;
        float freq = random_float(4.0f, 7.0f) * inv_size_factor;
        float amplidute = random_float(lx * 0.6f, lx * 1.2f);

        rivers.push_back({
            source_x, source_y,
            end_x, end_y,
            amplidute,
            freq,
            width,
            width,
            offsert
        });

        offsert += 500.0f;
    }
}

float RiverGenerator::river_factor(float x, float threshold, float height) {
    x = glm::clamp(x, 0.0f, 1.0f);

    // Precompute constants
    double A = (height - 1.0) / ((threshold - 1.0) * (threshold - 1.0));
    double a = 2.0 * (height - threshold) / (threshold * threshold * threshold * (threshold - 1.0));
    double b = (height * threshold - 3.0 * height + 2.0 * threshold) / (threshold * threshold * (threshold - 1.0));

    if (x <= threshold) {
        return a * x * x * x + b * x * x;
    } else {
        double d = x - 1.0;
        return A * d * d + 1.0;
    }
}

float RiverGenerator::calculate_single_river_distance(
    const perlin::Perlin2D& perlin_gen, 
    float x, float y, 
    int map_width, int map_height,
    const RiverProperties& props) 
{
    
    const float DETAIL_CONSTANT_D = 0.10f;
    const float HIGH_FREQ_SCALE = (0.1f);

    float sx = props.start_x;
    float sy = props.start_y;
    float ex = props.end_x;
    float ey = props.end_y;
    
    float dx = ex - sx;
    float dy = ey - sy;
    float lengthSq = dx * dx + dy * dy;
    
    float px = x - sx;
    float py = y - sy;
    
    float alpha = (px * dx + py * dy) / lengthSq;
    alpha = std::max(0.0f, std::min(1.0f, alpha));
    
    float p_straight_x = sx + alpha * dx;
    float p_straight_y = sy + alpha * dy;

    float N1_raw = perlin_gen.perlin_fractal(
        alpha * props.low_freq_scale + props.noise_offset, 
        0.0f, 
        4, 0.5f);
    
    float N1_centered = N1_raw - 0.5f;

    float perp_dx = -dy; 
    float perp_dy = dx;
    float perp_len = std::sqrt(perp_dx * perp_dx + perp_dy * perp_dy);

    float y_meander_offset_x = (perp_dx / perp_len) * N1_centered * props.meander_amplitude;
    float y_meander_offset_y = (perp_dy / perp_len) * N1_centered * props.meander_amplitude;

    float p_center_x = p_straight_x + y_meander_offset_x;
    float p_center_y = p_straight_y + y_meander_offset_y;

    float raw_distance_from_center = std::sqrt(
        std::pow(x - p_center_x, 2) + std::pow(y - p_center_y, 2)
    );
    
    float current_target_width_frac = props.taper_start_frac + (props.taper_end_frac - props.taper_start_frac) * alpha;
                                      
    float current_target_half_width = (float)map_height * current_target_width_frac / 2.0f;


    float normalized_distance = raw_distance_from_center / current_target_half_width;

    float N2 = perlin_gen.perlin_fractal(
        x * HIGH_FREQ_SCALE,
        y * HIGH_FREQ_SCALE,
        1, 0.5f);

    float river_value = normalized_distance + DETAIL_CONSTANT_D * N2 * 0.2f;

    return river_value;
}

float RiverGenerator::generate_river_value(
    const perlin::Perlin2D& perlin_gen, 
    float x, float y, 
    int map_width, int map_height,
    const std::vector<RiverProperties>& rivers) 
{
    float min_value = std::numeric_limits<float>::max();
    for (const auto& props : rivers) {
        min_value = std::min(min_value, calculate_single_river_distance(perlin_gen, x, y, map_width, map_height, props));
    }
    return std::min(min_value, 1.0f);
}

float Terrain::get_river_noise(int x, int y) {
    return RiverGenerator::river_factor(RiverGenerator::generate_river_value(*river_noise, x, y, lx, ly, rivers), 0.3f, 0.7f);
}

void Terrain::generate_rails(int stations) {
    int width = lx / GRID_CELL_SIZE;
    int height = ly / GRID_CELL_SIZE;

    glm::ivec2 P0(0.3f * width, 0.3f * height);
    glm::ivec2 P1(0.7f * width, 0.3f * height);
    glm::ivec2 P2(0.7f * width, 0.7f * height);
    glm::ivec2 P3(0.3f * width, 0.7f * height);

    std::vector<std::pair<glm::ivec2, glm::ivec2>> main_square = {{P0, P1}, {P1, P2}, {P2, P3}, {P3, P0}};

    for (auto segment : main_square) {
        add_rail(segment.first, segment.second, 30.0f);
    }

    int iter = 0;
    int cnt = 0;
    while (cnt < stations) {
        iter++;
        if (iter > 2000000) {
            break;
        }

        int station_x = random_int(0.1f * width, 0.9f * width);
        int station_y = random_int(0.1f * height, 0.9f * height);
        glm::vec2 station(station_x, station_y);

        bool good = true;

        for (int cx = station_x - START_AREA_SIZE; cx <= station_x + START_AREA_SIZE; cx++) {
            for (int cy = station_y - START_AREA_SIZE; cy <= station_y + START_AREA_SIZE; cy++) {
                float val = get_grid_cell(cx, cy);
                if (val < WATER_LEVEL + WATER_DELTA || val >= HILL_BEGIN) {
                    good = false;
                    break;
                }
            }
        }

        for (int i = 0; i < station_pos.size(); i++) {
            if (glm::length((glm::vec2)station - (glm::vec2)station_pos[i]) < 30.0f) {
                good = false;
                break;
            }
        }

        if (!good)
            continue;

        float tm = -1.0;
        int id = -1;
        float min_dist = 1000.0f;

        int i = 0;
        for (auto segment : main_square) {
            float t;
            float dist = distance_to_line2d(segment.first, segment.second, station, t);
            if (t >= -0.2f && t <= 1.2f && dist < min_dist) {
                min_dist = dist;
                id = i;
                tm = t;
            }

            i++;
        }

        if (min_dist < 10.0f || id == -1 || tm <= 0.1f || tm >= 0.9f)
            continue;

        float h = get_grid_cell(station_x, station_y);
        for (int cx = station_x - START_AREA_SIZE; cx <= station_x + START_AREA_SIZE; cx++) {
            for (int cy = station_y - START_AREA_SIZE; cy <= station_y + START_AREA_SIZE; cy++) {
                set_grid_cell(cx, cy, h);
            }
        }

        glm::ivec2 istation =  glm::ivec2(station_x, station_y);
        glm::ivec2 A = istation + glm::ivec2(-1, -1) * START_AREA_SIZE;
        glm::ivec2 B = istation + glm::ivec2(+1, -1) * START_AREA_SIZE;
        glm::ivec2 C = istation + glm::ivec2(+1, +1) * START_AREA_SIZE;
        glm::ivec2 D = istation + glm::ivec2(-1, +1) * START_AREA_SIZE;
        add_rail(A, B, 2 * START_AREA_SIZE + 1);
        add_rail(B, C, 2 * START_AREA_SIZE + 1);
        add_rail(C, D, 2 * START_AREA_SIZE + 1);
        add_rail(D, A, 2 * START_AREA_SIZE + 1);

        int line_x, line_y;
        if (id & 1) {
            line_x = main_square[id].first.x;
            line_y = station_y;
        } else {
            line_y = main_square[id].first.y;
            line_x = station_x;
        }

        glm::vec3 resource_pos = glm::vec3(station.x * GRID_CELL_SIZE + 0.5f * GRID_CELL_SIZE, h + 2.5f, station.y * GRID_CELL_SIZE + 0.5f * GRID_CELL_SIZE);

        station_pos.push_back(station);
        station_timers.push_back({true, 0.0f, resource_pos});

        if (cnt == 0) {
            resource_types.push_back(CENTRAL);
        } else {
            STATION_TYPE type = STATION_TYPE(1 + (cnt % 4));
            std::cout << type << std::endl;
            resource_types.push_back(type);
        }

        glm::ivec2 iline = glm::ivec2(line_x, line_y);
        glm::ivec2 dir = iline - istation;
        dir.x = glm::sign(dir.x);
        dir.y = glm::sign(dir.y);
        istation = istation - dir * (START_AREA_SIZE - 1);

        add_rail(iline, istation, 30.0f);

        cnt++;
    }

    std::cout << "Generated " << cnt << " stations" << std::endl;

    corner_values.resize(lx / GRID_CELL_SIZE + 1, std::vector<float>(ly / GRID_CELL_SIZE + 1, 0.0f));
    corner_power.resize(lx / GRID_CELL_SIZE + 1, std::vector<int>(ly / GRID_CELL_SIZE + 1, 0));

    for (int x = 0; x < lx / GRID_CELL_SIZE; x++) {
        for (int y = 0; y < ly / GRID_CELL_SIZE; y++) {
            if (rails[x][y] >= 0.0f) {
                corner_values[x][y] += rails[x][y];
                corner_values[x + 1][y] += rails[x][y];
                corner_values[x][y + 1] += rails[x][y];
                corner_values[x + 1][y + 1] += rails[x][y];

                corner_power[x][y]++;
                corner_power[x + 1][y]++;
                corner_power[x][y + 1]++;
                corner_power[x + 1][y + 1]++;
            }
        }
    }

    for (int x = 0; x <= lx / GRID_CELL_SIZE; x++) {
        for (int y = 0; y <= ly / GRID_CELL_SIZE; y++) {
            if (corner_power[x][y]) {
                corner_values[x][y] = corner_values[x][y] / corner_power[x][y];
            }
        }
    }
}

void Terrain::push_face(glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 D, glm::vec3 color, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    glm::vec3 norm = plane_normal(A, B, C);
    int vert_offset = vertices.size();

    vertices.push_back(VertexFormat(A, color, norm));
    vertices.push_back(VertexFormat(B, color, norm));
    vertices.push_back(VertexFormat(C, color, norm));
    vertices.push_back(VertexFormat(D, color, norm));

    indices.push_back(vert_offset + 1);
    indices.push_back(vert_offset + 0);
    indices.push_back(vert_offset + 2);

    indices.push_back(vert_offset + 2);
    indices.push_back(vert_offset + 0);
    indices.push_back(vert_offset + 3);
}

void Terrain::draw_station(int id, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    glm::ivec2 station = station_pos[id];

    float dd = START_AREA_SIZE - 1;
    glm::ivec2 dirs[] = {
        glm::ivec2(-dd, -dd),
        glm::ivec2(+dd, -dd),
        glm::ivec2(+dd, +dd),
        glm::ivec2(-dd, +dd),
    };

    float x_min = lx * 10.0f * GRID_CELL_SIZE;
    float y_min = ly * 10.0f * GRID_CELL_SIZE;
    float x_max = 0.0f;
    float y_max = 0.0f;

    float h = 6.0f;

    for (auto dir : dirs) {
        glm::ivec2 pos = station + dir;

        draw_pillar(pos.x, pos.y, h, 0.5f, vertices, indices);

        x_min = std::min(x_min, float(pos.x * GRID_CELL_SIZE));
        x_max = std::max(x_max, float(pos.x * GRID_CELL_SIZE + GRID_CELL_SIZE));

        y_min = std::min(y_min, float(pos.y * GRID_CELL_SIZE));
        y_max = std::max(y_max, float(pos.y * GRID_CELL_SIZE + GRID_CELL_SIZE));
    }

    float base_h = get_grid_cell_min(station.x, station.y);
    
    glm::vec3 A0 = glm::vec3(x_min, base_h + h + 0.5f, y_min);
    glm::vec3 B0 = glm::vec3(x_max, base_h + h + 0.5f, y_min);
    glm::vec3 C0 = glm::vec3(x_max, base_h + h + 0.5f, y_max);
    glm::vec3 D0 = glm::vec3(x_min, base_h + h + 0.5f, y_max);
    draw_slab(0.5f, A0, B0, C0, D0, glm::vec3(0.569, 0.447, 0.145), vertices, indices);

    glm::vec3 center = glm::vec3(station.x * GRID_CELL_SIZE + GRID_CELL_SIZE * 0.5f, base_h + h + 0.5f, station.y * GRID_CELL_SIZE + GRID_CELL_SIZE * 0.5f);

    glm::vec3 A1 = glm::vec3(x_min + GRID_CELL_SIZE, base_h + h + 0.5f, y_min + GRID_CELL_SIZE);
    glm::vec3 B1 = glm::vec3(x_max - GRID_CELL_SIZE, base_h + h + 0.5f, y_min + GRID_CELL_SIZE);
    glm::vec3 C1 = glm::vec3(x_max - GRID_CELL_SIZE, base_h + h + 0.5f, y_max - GRID_CELL_SIZE);
    glm::vec3 D1 = glm::vec3(x_min + GRID_CELL_SIZE, base_h + h + 0.5f, y_max - GRID_CELL_SIZE);

    switch (resource_types[id])
    {
    case CENTRAL: {
        draw_semisphere(center, (x_max - x_min) * 0.5f, vertices, indices, glm::vec3(0.914, 0.184, 0.961));
        draw_semisphere(A1, GRID_CELL_SIZE, vertices, indices, glm::vec3(0.914, 0.184, 0.961));
        draw_semisphere(B1, GRID_CELL_SIZE, vertices, indices, glm::vec3(0.914, 0.184, 0.961));
        draw_semisphere(C1, GRID_CELL_SIZE, vertices, indices, glm::vec3(0.914, 0.184, 0.961));
        draw_semisphere(D1, GRID_CELL_SIZE, vertices, indices, glm::vec3(0.914, 0.184, 0.961));
    } break;

    case WOOD: {
        draw_circular_pyramid(center, (x_max - x_min) * 0.5f, 1.5f * h, vertices, indices, glm::vec3(0.29f, 1, 0.463f));
        draw_circular_pyramid(A1, GRID_CELL_SIZE, 0.5f * h, vertices, indices, glm::vec3(0.29f, 1, 0.463f));
        draw_circular_pyramid(B1, GRID_CELL_SIZE, 0.5f * h, vertices, indices, glm::vec3(0.29f, 1, 0.463f));
        draw_circular_pyramid(C1, GRID_CELL_SIZE, 0.5f * h, vertices, indices, glm::vec3(0.29f, 1, 0.463f));
        draw_circular_pyramid(D1, GRID_CELL_SIZE, 0.5f * h, vertices, indices, glm::vec3(0.29f, 1, 0.463f));
    } break;

    case STONE: {
        float radius = (x_max - x_min) * 0.5f;
        float dh = 1.0f;

        glm::vec3 c = center;

        for (int i = 0; i < 6; i++) {
            draw_cylinder(c, radius, 1.0f, vertices, indices, glm::vec3(1, 0.392, 0.102));
            radius -= 1.5f;
            c.y += dh;
        }

        radius += 1.5f;
        draw_circular_pyramid(c, radius, 3.0f, vertices, indices, glm::vec3(1, 0.392, 0.102));
    } break;

    case COAL: {
        float step = 1.0f;
        glm::vec3 points[] = {A0, B0, C0, D0};
        glm::vec3 deltas[] = {
            glm::vec3(+GRID_CELL_SIZE * 0.3f, step, +GRID_CELL_SIZE * 0.3f),
            glm::vec3(-GRID_CELL_SIZE * 0.3f, step, +GRID_CELL_SIZE * 0.3f),
            glm::vec3(-GRID_CELL_SIZE * 0.3f, step, -GRID_CELL_SIZE * 0.3f),
            glm::vec3(+GRID_CELL_SIZE * 0.3f, step, -GRID_CELL_SIZE * 0.3f)
        };

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 4; j++) {
                points[j] += deltas[j]; 
            }

            draw_slab(step, points[0], points[1], points[2], points[3], glm::vec3(0.322, 0.506, 0.812), vertices, indices);
        }

        center.y = points[0].y;
        draw_square_pyramid(center, (points[1].x - points[0].x) * 0.5f, 3.0f, vertices, indices, glm::vec3(0.322, 0.506, 0.812));
    } break;

    case GOLD: {
        draw_square_pyramid(center, (x_max - x_min) * 0.5f, 1.5f * h, vertices, indices, glm::vec3(0.961f, 0.925f, 0.259f));
        draw_square_pyramid(A1, GRID_CELL_SIZE, 1.0f * h, vertices, indices, glm::vec3(0.961f, 0.925f, 0.259f));
        draw_square_pyramid(B1, GRID_CELL_SIZE, 1.0f * h, vertices, indices, glm::vec3(0.961f, 0.925f, 0.259f));
        draw_square_pyramid(C1, GRID_CELL_SIZE, 1.0f * h, vertices, indices, glm::vec3(0.961f, 0.925f, 0.259f));
        draw_square_pyramid(D1, GRID_CELL_SIZE, 1.0f * h, vertices, indices, glm::vec3(0.961f, 0.925f, 0.259f));
    } break;

    default:
        break;
    }
}

void Terrain::draw_pillar(int x, int y, float h, float t, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    float width = t * GRID_CELL_SIZE;
    float delta = (GRID_CELL_SIZE - width) * 0.5f;
    
    float x_min = x * GRID_CELL_SIZE + delta;
    float y_min = y * GRID_CELL_SIZE + delta;
    float x_max = x_min + GRID_CELL_SIZE - 2.0f * delta;
    float y_max = y_min + GRID_CELL_SIZE - 2.0f * delta;

    float base_h = get_grid_cell_min(x, y);
    
    glm::vec3 A0 = glm::vec3(x_min, base_h + h, y_min);
    glm::vec3 B0 = glm::vec3(x_max, base_h + h, y_min);
    glm::vec3 C0 = glm::vec3(x_max, base_h + h, y_max);
    glm::vec3 D0 = glm::vec3(x_min, base_h + h, y_max);

    draw_slab(h, A0, B0, C0, D0, glm::vec3(0.569, 0.447, 0.145), vertices, indices);
}

void Terrain::draw_rail_cell(int x, int y, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices, bool draw_tunnel)
{
    if (rails[x][y] < 0.0f)
        return;

    glm::vec3 up(0.0f, 1.0f, 0.0f);
    float delta_h = 0.5f;

    float real_value = rails[x][y];
    float value = get_grid_cell(x, y);

    float x_min = x * GRID_CELL_SIZE;
    float y_min = y * GRID_CELL_SIZE;
    float x_max = x_min + GRID_CELL_SIZE;
    float y_max = y_min + GRID_CELL_SIZE;

    float ph[] = {corner_values[x][y], corner_values[x + 1][y], corner_values[x + 1][y + 1], corner_values[x][y + 1]};

    glm::vec3 A0 = glm::vec3(x_min, ph[0], y_min);
    glm::vec3 B0 = glm::vec3(x_max, ph[1], y_min);
    glm::vec3 C0 = glm::vec3(x_max, ph[2], y_max);
    glm::vec3 D0 = glm::vec3(x_min, ph[3], y_max);

    glm::vec3 A1 = A0 + delta_h * up;
    glm::vec3 B1 = B0 + delta_h * up; 
    glm::vec3 C1 = C0 + delta_h * up; 
    glm::vec3 D1 = D0 + delta_h * up;

    glm::vec3 color = glm::vec3(0.3f, 0.3f, 0.35f);

    if (!draw_tunnel) {
        push_face(A1, B1, C1, D1, color, vertices, indices);
        return;
    }

    draw_slab(3.5f * delta_h, A1, B1, C1, D1, color, vertices, indices);

    if (real_value - value >= 1.0f) {
        // BRIDGE
        color += glm::vec3(0.4f);
        glm::ivec2 dirs[] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
        std::vector<std::vector<glm::vec3>> base = {{A1, B1}, {B1, C1}, {C1, D1}, {D1, A1}};

        for (int i = 0; i < 4; i++) {
            float nx = x + dirs[i].x;
            float ny = y + dirs[i].y;
            glm::vec3 dir = glm::vec3(dirs[i].x, 0.0f, dirs[i].y);

            if (rails[nx][ny] < 0.0f) {
                glm::vec3 AA0 = base[i][0] + 1.0f * up;
                glm::vec3 BB0 = base[i][1] + 1.0f * up;
                glm::vec3 CC0 = BB0 + 0.5f * dir;
                glm::vec3 DD0 = AA0 + 0.5f * dir;
                draw_slab(1.0f, AA0, DD0, CC0, BB0, color, vertices, indices);
            }
        }
    }

    if (value - real_value >= 1.0f) {
        // TUNEL
        color += glm::vec3(0.1f);
        float tunnel_str = 1.5f;

        glm::vec3 A2 = A1 + (4.0f + tunnel_str) * up;
        glm::vec3 B2 = B1 + (4.0f + tunnel_str) * up; 
        glm::vec3 C2 = C1 + (4.0f + tunnel_str) * up; 
        glm::vec3 D2 = D1 + (4.0f + tunnel_str) * up;

        draw_slab(tunnel_str, A2, B2, C2, D2, color, vertices, indices);

        glm::ivec2 dirs[] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
        std::vector<std::vector<glm::vec3>> base = {{A1, B1}, {B1, C1}, {C1, D1}, {D1, A1}};

        for (int i = 0; i < 4; i++) {
            float nx = x + dirs[i].x;
            float ny = y + dirs[i].y;
            glm::vec3 dir = glm::vec3(dirs[i].x, 0.0f, dirs[i].y);

            if (rails[nx][ny] < 0.0f) {
                glm::vec3 AA0 = base[i][0] + 4.0f * up;
                glm::vec3 BB0 = base[i][1] + 4.0f * up;
                glm::vec3 CC0 = BB0 - 0.5f * dir;
                glm::vec3 DD0 = AA0 - 0.5f * dir;
                draw_slab(4.0f, AA0, BB0, CC0, DD0, color, vertices, indices);
            }
        }
    }
}

void Terrain::draw_slab(float h, glm::vec3 A1, glm::vec3 B1, glm::vec3 C1, glm::vec3 D1, glm::vec3 color, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    glm::vec3 A2 = A1 - h * up;
    glm::vec3 B2 = B1 - h * up; 
    glm::vec3 C2 = C1 - h * up; 
    glm::vec3 D2 = D1 - h * up; 

    push_face(A1, B1, C1, D1, color, vertices, indices);
    push_face(A2, D2, C2, B2, color, vertices, indices);

    push_face(C2, C1, B1, B2, color, vertices, indices);
    push_face(B2, B1, A1, A2, color, vertices, indices);
    push_face(A2, A1, D1, D2, color, vertices, indices);
    push_face(D2, D1, C1, C2, color, vertices, indices); 
}

float Terrain::compute_rail_height(int x, int y, float desired_height, bool &modify) {
    float h = get_grid_cell(x, y);
    float delta = h - desired_height;

    if (h >= WATER_LEVEL + WATER_DELTA && delta <= 5.0f) {
        modify = true;

        if (desired_height <= WATER_LEVEL + WATER_DELTA) {
            return WATER_LEVEL + WATER_DELTA;
        }

        return desired_height;
    }

    modify = false;
    return h;
}

void Terrain::add_rail(glm::ivec2 P0, glm::ivec2 P1, int step_size) {
    add_rail(P0.x, P0.y, P1.x, P1.y, step_size);
}

void Terrain::add_rail(int source_x, int source_y, int end_x, int end_y, int step_size)
{
    // Determine direction
    int dx = (end_x > source_x) ? 1 : -1;
    int dy = (end_y > source_y) ? 1 : -1;

    // Pivot always at (end_x, source_y)
    int pivot_x = end_x;
    int pivot_y = source_y;

    // Height interpolation endpoints
    float h0 = get_grid_cell(source_x, source_y);
    if (rails[source_x][source_y] >= 0.0f)
        h0 = rails[source_x][source_y];

    float h1 = get_grid_cell(end_x, end_y);
    if (rails[end_x][end_y] >= 0.0f)
        h1 = rails[end_x][end_y];

    float hp = get_grid_cell(pivot_x, pivot_y);
    if (rails[pivot_x][pivot_y] >= 0.0f)
        hp = rails[pivot_x][pivot_y];

    int x = source_x;
    while (x != pivot_x) {
        int section_end_x = x + dx * step_size;

        // Clamp to pivot
        if ((dx > 0 && section_end_x + dx * step_size > pivot_x) ||
            (dx < 0 && section_end_x + dx * step_size < pivot_x))
            section_end_x = pivot_x;

        // Stop if hitting existing rail
        for (int cx = x + dx; cx != section_end_x + dx; cx += dx) {
            if (rails[cx][pivot_y] >= 0.0f) {
                section_end_x = cx;
                break;
            }
        }

        flat_for_rail(x, pivot_y, section_end_x, pivot_y);
        x = section_end_x;
    }

    int y = pivot_y;
    while (y != end_y) {
        int section_end_y = y + dy * step_size;

        // Clamp to goal
        if ((dy > 0 && section_end_y + dy * step_size > end_y) ||
            (dy < 0 && section_end_y + dy * step_size < end_y))
            section_end_y = end_y;

        // Stop if hitting existing rail
        for (int cy = y + dy; cy != section_end_y + dy; cy += dy) {
            if (rails[pivot_x][cy] >= 0.0f) {
                section_end_y = cy;
                break;
            }
        }

        flat_for_rail(pivot_x, y, pivot_x, section_end_y);
        y = section_end_y;
    }
}

void Terrain::flat_for_rail(int x0, int y0, int x1, int y1) {
    auto get_h = [&](int x, int y) {
        float h = get_grid_cell(x, y);
        if (rails[x][y] >= 0.0f)
            h = rails[x][y];
        return h;
    };

    float h0 = get_h(x0, y0);
    float h1 = get_h(x1, y1);

    std::vector<float> backup;

    if (y0 == y1)
    {
        int y = y0;
        int xmin = std::min(x0, x1);
        int xmax = std::max(x0, x1);

        for (int x = xmin; x <= xmax; x++) {

            float denom = float(x1 - x0);
            float t = (std::abs(denom) < 1e-6f) ? 0.0f : (x - x0) / denom;

            float desired_height = (1.0f - t) * h0 + t * h1;

            bool modif = false;
            float final_height = compute_rail_height(x, y, desired_height, modif);

            backup.push_back(modif ? final_height : -1.0f);

            desired_height = std::max(desired_height, WATER_LEVEL + WATER_DELTA);
            rails[x][y] = desired_height;
        }

        // Write terrain changes
        int idx = 0;
        for (int x = xmin; x <= xmax; x++, idx++)
            if (backup[idx] >= 0.0f)
                set_grid_cell(x, y, backup[idx]);

        return;
    }

    if (x0 == x1)
    {
        int x = x0;
        int ymin = std::min(y0, y1);
        int ymax = std::max(y0, y1);

        for (int y = ymin; y <= ymax; y++) {

            float denom = float(y1 - y0);
            float t = (std::abs(denom) < 1e-6f) ? 0.0f : (y - y0) / denom;

            float desired_height = (1.0f - t) * h0 + t * h1;

            bool modif = false;
            float final_height = compute_rail_height(x, y, desired_height, modif);

            backup.push_back(modif ? final_height : -1.0f);

            desired_height = std::max(desired_height, WATER_LEVEL + WATER_DELTA);
            rails[x][y] = desired_height;
        }

        // Write terrain changes
        int idx = 0;
        for (int y = ymin; y <= ymax; y++, idx++)
            if (backup[idx] >= 0.0f)
                set_grid_cell(x, y, backup[idx]);

        return;
    }

    return;
}

void Terrain::create_grid() {
    grid.resize(lx / GRID_CELL_SIZE, std::vector<grid_cell>(ly / GRID_CELL_SIZE));

    for (int x = 0; x < lx / GRID_CELL_SIZE; x++) {
        for (int y = 0; y < ly / GRID_CELL_SIZE; y++) {
            grid[x][y].height = rails[x][y];

            if (rails[x][y] < 0.0f) {
                grid[x][y].norm = glm::vec3(0.0f, 1.0f, 0.0f);
                continue;
            }

            float x_min = x * GRID_CELL_SIZE;
            float y_min = y * GRID_CELL_SIZE;
            float x_max = x_min + GRID_CELL_SIZE;
            float y_max = y_min + GRID_CELL_SIZE;

            float ph[] = {corner_values[x][y], corner_values[x + 1][y], corner_values[x + 1][y + 1], corner_values[x][y + 1]};

            glm::vec3 A0 = glm::vec3(x_min, ph[0], y_min);
            glm::vec3 B0 = glm::vec3(x_max, ph[1], y_min);
            glm::vec3 C0 = glm::vec3(x_max, ph[2], y_max);
            glm::vec3 D0 = glm::vec3(x_min, ph[3], y_max);

            glm::vec3 n1 = plane_normal(A0, B0, C0);
            glm::vec3 n2 = plane_normal(B0, C0, D0);
            glm::vec3 n3 = plane_normal(C0, D0, A0);
            glm::vec3 n4 = plane_normal(D0, A0, B0);

            glm::vec3 aprox_norm = glm::normalize(n1 + n2 + n3 + n4);

            grid[x][y].norm = aprox_norm;
        }
    }
}

grid_cell Terrain::get_on_train_grid(int x, int y) {
    return grid[x][y];
}

std::vector<glm::ivec2> *Terrain::get_stations() {
    return &station_pos;
}

float Terrain::get_grid_cell_min(int x, int y) {
    float value = 1000.0f;

    for (int sx = x * GRID_CELL_SIZE; sx < x * GRID_CELL_SIZE + GRID_CELL_SIZE; sx++) {
        for (int sy = y * GRID_CELL_SIZE; sy < y * GRID_CELL_SIZE + GRID_CELL_SIZE; sy++) {
            value = std::min(
                heightmap[sx][sy],
                std::min(
                    heightmap[sx + 1][sy],
                    std::min(
                        heightmap[sx][sy + 1],
                        std::min(
                            heightmap[sx + 1][sy + 1],
                            value
                        )
                    )
                )
            );
        }
    }

    return value;
}

bool Terrain::is_tunnel_cell(int x, int y)
{
    float value = get_grid_cell(x, y);
    float real_value = rails[x][y];
    return value - real_value >= 1.0f;
}

void Terrain::generate_under_terrain_mesh(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    for (int x = 0; x < lx / GRID_CELL_SIZE; x++) {
        for (int y = 0; y < ly / GRID_CELL_SIZE; y++) {
            //if (is_tunnel_cell(x, y))
            draw_rail_cell(x, y, vertices, indices, false);
        }
    }
}

void Terrain::draw_semisphere(glm::vec3 center, float radius,
                              std::vector<VertexFormat> &vertices,
                              std::vector<unsigned int> &indices,
                              glm::vec3 color)
{
    const int stacks = 24;
    const int slices = 48;

    unsigned int baseIndex = vertices.size();

    for (int i = 0; i <= stacks; ++i) {
        float phi = (float)i / (float)stacks * (glm::pi<float>() * 0.5f);

        for (int j = 0; j <= slices; ++j) {
            float theta = (float)j / (float)slices * glm::two_pi<float>();

            float x = radius * sin(phi) * cos(theta);
            float y = radius * cos(phi);
            float z = radius * sin(phi) * sin(theta);

            glm::vec3 pos = center + glm::vec3(x, y, z);
            glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));

            vertices.emplace_back(pos, color, normal);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            unsigned int row1 = baseIndex + i * (slices + 1) + j;
            unsigned int row2 = baseIndex + (i + 1) * (slices + 1) + j;

            // two triangles per quad
            indices.push_back(row2);
            indices.push_back(row1);
            indices.push_back(row1 + 1);

            indices.push_back(row2);
            indices.push_back(row1 + 1);
            indices.push_back(row2 + 1);
        }
    }
}

void Terrain::draw_square_pyramid(glm::vec3 center, float size, float height,
                                  std::vector<VertexFormat> &vertices,
                                  std::vector<unsigned int> &indices,
                                  glm::vec3 color)
{
    glm::vec3 A0 = center + glm::vec3(-size, 0.0f, -size);
    glm::vec3 B0 = center + glm::vec3(+size, 0.0f, -size);
    glm::vec3 C0 = center + glm::vec3(+size, 0.0f, +size);
    glm::vec3 D0 = center + glm::vec3(-size, 0.0f, +size);

    glm::vec3 H = center + glm::vec3(0.0f, height, 0.0f);

    unsigned int baseIndex = vertices.size();

    glm::vec3 baseNormal = glm::vec3(0.0f, -1.0f, 0.0f);
    vertices.push_back(VertexFormat(A0, color, baseNormal));
    vertices.push_back(VertexFormat(B0, color, baseNormal));
    vertices.push_back(VertexFormat(C0, color, baseNormal));
    vertices.push_back(VertexFormat(D0, color, baseNormal));

    indices.push_back(baseIndex + 0);
    indices.push_back(baseIndex + 1);
    indices.push_back(baseIndex + 2);

    indices.push_back(baseIndex + 0);
    indices.push_back(baseIndex + 2);
    indices.push_back(baseIndex + 3);

    auto compute_normal = [](glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        return glm::normalize(glm::cross(b - a, c - a));
    };

    glm::vec3 normal;

    normal = compute_normal(A0, H, B0);
    vertices.push_back(VertexFormat(A0, color, normal));
    vertices.push_back(VertexFormat(B0, color, normal));
    vertices.push_back(VertexFormat(H, color, normal));
    indices.push_back(baseIndex + 4);
    indices.push_back(baseIndex + 6);
    indices.push_back(baseIndex + 5);

    normal = compute_normal(B0, H, C0);
    vertices.push_back(VertexFormat(B0, color, normal));
    vertices.push_back(VertexFormat(C0, color, normal));
    vertices.push_back(VertexFormat(H, color, normal));
    indices.push_back(baseIndex + 7);
    indices.push_back(baseIndex + 9);
    indices.push_back(baseIndex + 8);

    normal = compute_normal(C0, H, D0);
    vertices.push_back(VertexFormat(C0, color, normal));
    vertices.push_back(VertexFormat(D0, color, normal));
    vertices.push_back(VertexFormat(H, color, normal));
    indices.push_back(baseIndex + 10);
    indices.push_back(baseIndex + 12);
    indices.push_back(baseIndex + 11);

    normal = compute_normal(D0, H, A0);
    vertices.push_back(VertexFormat(D0, color, normal));
    vertices.push_back(VertexFormat(A0, color, normal));
    vertices.push_back(VertexFormat(H, color, normal));
    indices.push_back(baseIndex + 13);
    indices.push_back(baseIndex + 15);
    indices.push_back(baseIndex + 14);
}

void Terrain::draw_circular_pyramid(glm::vec3 center, float radius, float height,
                                    std::vector<VertexFormat> &vertices,
                                    std::vector<unsigned int> &indices,
                                    glm::vec3 color)
{
    const int segments = 36;
    unsigned int baseIndex = vertices.size();

    glm::vec3 apex = center + glm::vec3(0.0f, height, 0.0f);

    for (int i = 0; i < segments; ++i) {
        float theta1 = (float)i / segments * glm::two_pi<float>();
        float theta2 = (float)(i + 1) / segments * glm::two_pi<float>();

        glm::vec3 v1 = center + glm::vec3(radius * cos(theta1), 0.0f, radius * sin(theta1));
        glm::vec3 v2 = center + glm::vec3(radius * cos(theta2), 0.0f, radius * sin(theta2));

        glm::vec3 normal = glm::normalize(glm::cross(v2 - apex, v1 - apex));

        vertices.push_back(VertexFormat(v1, color, normal));
        vertices.push_back(VertexFormat(apex, color, normal));
        vertices.push_back(VertexFormat(v2, color, normal));

        indices.push_back(baseIndex++);
        indices.push_back(baseIndex++);
        indices.push_back(baseIndex++);
    }
}

void Terrain::draw_cylinder(glm::vec3 center, float radius, float height,
                            std::vector<VertexFormat> &vertices,
                            std::vector<unsigned int> &indices,
                            glm::vec3 color)
{
    const int segments = 36;
    unsigned int baseIndex = vertices.size();

    glm::vec3 topCenter = center + glm::vec3(0.0f, height, 0.0f);
    glm::vec3 topNormal = glm::vec3(0.0f, 1.0f, 0.0f);

    vertices.push_back(VertexFormat(topCenter, color, topNormal));
    unsigned int topCenterIndex = baseIndex;

    std::vector<glm::vec3> topVertices;
    std::vector<glm::vec3> bottomVertices;
    for (int i = 0; i < segments; ++i) {
        float theta = (float)i / segments * glm::two_pi<float>();
        float x = radius * cos(theta);
        float z = radius * sin(theta);

        glm::vec3 topPos = topCenter + glm::vec3(x, 0.0f, z);
        glm::vec3 bottomPos = center + glm::vec3(x, 0.0f, z);

        topVertices.push_back(topPos);
        bottomVertices.push_back(bottomPos);

        vertices.push_back(VertexFormat(topPos, color, topNormal));
    }

    for (int i = 0; i < segments; ++i) {
        int next = (i + 1) % segments;
        indices.push_back(baseIndex + 1 + i);
        indices.push_back(topCenterIndex);
        indices.push_back(baseIndex + 1 + next);
    }

    for (int i = 0; i < segments; ++i) {
        int next = (i + 1) % segments;

        glm::vec3 v0 = bottomVertices[i];
        glm::vec3 v1 = bottomVertices[next];
        glm::vec3 v2 = topVertices[next];
        glm::vec3 v3 = topVertices[i];

        glm::vec3 normal = -glm::normalize(glm::cross(v1 - v0, v3 - v0));

        vertices.push_back(VertexFormat(v0, color, normal));
        vertices.push_back(VertexFormat(v3, color, normal));
        vertices.push_back(VertexFormat(v2, color, normal));
        vertices.push_back(VertexFormat(v1, color, normal));

        unsigned int idx = vertices.size();
        indices.push_back(idx - 4);
        indices.push_back(idx - 3);
        indices.push_back(idx - 2);
        indices.push_back(idx - 2);
        indices.push_back(idx - 1);
        indices.push_back(idx - 4);
    }
}

void Terrain::generate_request(int cnt) {
    request.clear();
    for (int i = 0; i < cnt; i++) {
        int type = 1 + random_int(0, 3);
        request.push_back(STATION_TYPE(type));
    }
}

void Terrain::handle_train(glm::ivec2 train_pos) {
    for (int i = 1; i < station_pos.size(); i++) {
        if (train_pos != station_pos[i])
            continue;

        auto it = std::find(request.begin(), request.end(), resource_types[i]);

        if (station_timers[i].has_resource && it != request.end()) {
            request.erase(it);
            station_timers[i].has_resource = false;
        }
    }

    if (train_pos == station_pos[0] && request.size() == 0) {
        mission_time = 0.0f;
        generate_request(5);
    }
}

void Terrain::update_stations(float delta_time)
{
    if (game_over)
        return;

    for (int i = 1; i < station_timers.size(); i++) {
        station_timers[i].update(delta_time);
    }

    if (mission_time >= MISSION_TIMEOUT) {
        // GAME OVER
        game_over = true;
    }

    mission_time += delta_time;
    internal_time += delta_time;
}

void Terrain::draw(draw_mesh_func draw_func) {
    float h_delta = glm::sin(internal_time * 5.0f);
    
    for (int i = 1; i < station_timers.size(); i++) {
        if (station_timers[i].has_resource) {
            glm::mat4 model = glm::mat4(1.0f);

            model = glm::translate(model, station_timers[i].resource_pos + glm::vec3(0, h_delta, 0));
            model = glm::rotate(model, internal_time * 5.0f, glm::vec3(0, 1, 0));
            model = glm::scale(model, glm::vec3(1.5f));

            int resource_id = resource_types[i] - 1;

            draw_func(("resource_" + std::to_string(resource_id)).c_str(), model, 1.0f);
        }
    }

    if (request.size() == 0) {
        glm::mat4 model = glm::mat4(1.0f);

        model = glm::translate(model, station_timers[0].resource_pos + glm::vec3(0, 3.0f * h_delta + 30.0f, 0));
        model = glm::rotate(model, internal_time * 5.0f, glm::vec3(0, 1, 0));
        model = glm::scale(model, glm::vec3(3.0f, 15.0f, 3.0f));

        draw_func("ping", model, 1.0f);
    }
}

bool Terrain::is_gameover() {
    return game_over;
}

float Terrain::get_mission_time() {
    return mission_time;
}

int Terrain::request_size() {
    return request.size();
}

void Terrain::draw_ui(draw_mesh_func draw_func, int h, int w) {
    float x_init = 15.0f;

    for (int i = 0; i < request.size(); i++) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(x_init, 0.0f, 30.0f));
        model = glm::scale(model, glm::vec3(10.0f));
        
        int resource_id = request[i] - 1;
        draw_func(("resource_" + std::to_string(resource_id)).c_str(), model, 1.0f);

        x_init += 20.0f;
    }
}

void Resources::draw_cube(glm::vec3 center, float half_size, glm::vec3 color, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    unsigned int base = vertices.size();
    
    glm::vec3 v[8] = {
        center + glm::vec3(-half_size, -half_size, -half_size),
        center + glm::vec3(+half_size, -half_size, -half_size),
        center + glm::vec3(+half_size, +half_size, -half_size),
        center + glm::vec3(-half_size, +half_size, -half_size),
        center + glm::vec3(-half_size, -half_size, +half_size),
        center + glm::vec3(+half_size, -half_size, +half_size),
        center + glm::vec3(+half_size, +half_size, +half_size),
        center + glm::vec3(-half_size, +half_size, +half_size)  
    };
    
    glm::vec3 normalFront  = glm::vec3(0, 0, -1);
    glm::vec3 normalBack   = glm::vec3(0, 0, 1);
    glm::vec3 normalLeft   = glm::vec3(-1, 0, 0);
    glm::vec3 normalRight  = glm::vec3(1, 0, 0); 
    glm::vec3 normalTop    = glm::vec3(0, 1, 0);
    glm::vec3 normalBottom = glm::vec3(0, -1, 0);
    
    vertices.push_back(VertexFormat(v[0], color, normalFront));
    vertices.push_back(VertexFormat(v[1], color, normalFront));
    vertices.push_back(VertexFormat(v[2], color, normalFront));
    vertices.push_back(VertexFormat(v[3], color, normalFront));
    
    vertices.push_back(VertexFormat(v[4], color, normalBack));
    vertices.push_back(VertexFormat(v[5], color, normalBack));
    vertices.push_back(VertexFormat(v[6], color, normalBack));
    vertices.push_back(VertexFormat(v[7], color, normalBack));
    
    vertices.push_back(VertexFormat(v[0], color, normalLeft));
    vertices.push_back(VertexFormat(v[3], color, normalLeft));
    vertices.push_back(VertexFormat(v[7], color, normalLeft));
    vertices.push_back(VertexFormat(v[4], color, normalLeft));
    
    vertices.push_back(VertexFormat(v[1], color, normalRight));
    vertices.push_back(VertexFormat(v[5], color, normalRight));
    vertices.push_back(VertexFormat(v[6], color, normalRight));
    vertices.push_back(VertexFormat(v[2], color, normalRight));
    
    vertices.push_back(VertexFormat(v[3], color, normalTop));
    vertices.push_back(VertexFormat(v[2], color, normalTop));
    vertices.push_back(VertexFormat(v[6], color, normalTop));
    vertices.push_back(VertexFormat(v[7], color, normalTop));
    
    vertices.push_back(VertexFormat(v[0], color, normalBottom));
    vertices.push_back(VertexFormat(v[4], color, normalBottom));
    vertices.push_back(VertexFormat(v[5], color, normalBottom));
    vertices.push_back(VertexFormat(v[1], color, normalBottom));

    indices.insert(indices.end(), {base+0, base+2, base+1, base+0, base+3, base+2});
    indices.insert(indices.end(), {base+5, base+7, base+4, base+5, base+6, base+7});
    indices.insert(indices.end(), {base+11, base+9, base+8, base+11, base+10, base+9});
    indices.insert(indices.end(), {base+12, base+14, base+13, base+12, base+15, base+14});
    indices.insert(indices.end(), {base+19, base+17, base+16, base+19, base+18, base+17});
    indices.insert(indices.end(), {base+20, base+22, base+21, base+20, base+23, base+22});
}

void Resources::create_resourve(int i, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    switch (i)
    {
    case 0:
        wood_resource(vertices, indices);
        break;
    case 1:
        stone_resource(vertices, indices);
        break;
    case 2:
        coal_resource(vertices, indices);
        break;
    case 3:
        gold_resource(vertices, indices);
        break;
    
    default:
        break;
    }
}

void Resources::wood_resource(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    draw_cube(glm::vec3(0.0f), 0.5f, glm::vec3(0.749, 0.502, 0.122), vertices, indices);
}

void Resources::stone_resource(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    draw_cube(glm::vec3(0.0f), 0.5f, glm::vec3(0.631, 0.604, 0.569), vertices, indices);
}

void Resources::coal_resource(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    draw_cube(glm::vec3(0.0f), 0.5f, glm::vec3(0.251, 0.247, 0.239), vertices, indices);
}

void Resources::gold_resource(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices) {
    draw_cube(glm::vec3(0.0f), 0.5f, glm::vec3(0.988, 0.976, 0.212), vertices, indices);
}

void station_data::update(float delta_dime) {
    if (has_resource)
        return;

    time += delta_dime;
    if (time >= RESOURCE_REGENERATION_TIME) {
        time = 0.0f;
        has_resource = true;
    }
}
