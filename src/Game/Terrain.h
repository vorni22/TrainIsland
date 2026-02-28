#pragma once

#include <vector>
#include <functional>
#include <unordered_map>

#include "PerlinNoise.h"
#include "components/simple_scene.h"

#define GRID_CELL_SIZE 4
#define WATER_LEVEL 5.0f
#define WATER_DELTA 1.0f
#define HILL_BEGIN 20.0f
#define START_AREA_SIZE 3
#define RESOURCE_REGENERATION_TIME 5.0f
#define MISSION_TIMEOUT 120.0f

typedef std::function<void(const char *, const std::vector<VertexFormat>&, const std::vector<unsigned int>&)> create_mesh_func;
typedef std::function<void(const char *, const glm::mat4&, float)> draw_mesh_func;

enum STATION_TYPE {
    CENTRAL = 0,
    WOOD = 1,
    STONE = 2,
    COAL = 3,
    GOLD = 4,
};

struct grid_cell {
    float height;
    glm::vec3 norm;
};

struct station_data {
    bool has_resource;
    float time;
    glm::vec3 resource_pos;

    void update(float delta_dime);
};

class Resources {
public:
    static void draw_cube(glm::vec3 center, float half_size, glm::vec3 color, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);

    static void create_resourve(int i, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);

    static void wood_resource(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);
    static void stone_resource(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);
    static void coal_resource(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);
    static void gold_resource(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);
};

struct RiverProperties {
    float start_x, start_y;      // Map coordinates for the start (source)
    float end_x, end_y;          // Map coordinates for the end (mouth)
    float meander_amplitude;     // Controls the side-to-side wiggle.
    float low_freq_scale;        // Controls the length of the meander curves.
    float taper_start_frac;      // River width (as fraction of map height) at start.
    float taper_end_frac;        // River width (as fraction of map height) at end.
    float noise_offset;          // Offset for the Perlin noise to ensure unique paths.
};

class RiverGenerator {
public:
    static float river_factor(float x, float threshold, float height);

    static float calculate_single_river_distance(
        const perlin::Perlin2D& perlin_gen, 
        float x, float y, 
        int map_width, int map_height,
        const RiverProperties& props
    );

    static float generate_river_value(
        const perlin::Perlin2D& perlin_gen, 
        float x, float y, 
        int map_width, int map_height,
        const std::vector<RiverProperties>& rivers
    );
};

class Terrain {
public:
    Terrain(int lenx, int leny, create_mesh_func create_mesh);

    void generate_mesh(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);
    void generate_rail_mesh(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);
    void generate_under_terrain_mesh(std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);

    void set_grid_cell(int x, int y, float value);
    float get_grid_cell(int x, int y);
    float get_grid_cell_min(int x, int y);

    grid_cell get_on_train_grid(int x, int y);
    bool is_tunnel_cell(int x, int y);

    std::vector<glm::ivec2> *get_stations();

    void handle_train(glm::ivec2 train_pos);
    void update_stations(float delta_time);
    void draw(draw_mesh_func draw_func);

    bool is_gameover();

    float get_mission_time();
    int request_size();

    void draw_ui(draw_mesh_func draw_func, int h, int w);

private:
    void generate_rivers(int cnt);
    float get_river_noise(int x, int y);

    void generate_rails(int stations);

    void draw_station(int id, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);
    void draw_pillar(int x, int y, float h, float t, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);

    void push_face(glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 D, glm::vec3 color, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);
    void draw_rail_cell(int x, int y, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices,  bool draw_tunnel);
    void draw_slab(float h, glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 D, glm::vec3 color, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices);

    void draw_semisphere(glm::vec3 center, float radius, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices, glm::vec3 color);
    void draw_square_pyramid(glm::vec3 center, float size, float height, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices, glm::vec3 color);
    void draw_circular_pyramid(glm::vec3 center, float radius, float height, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices, glm::vec3 color);
    void draw_cylinder(glm::vec3 center, float radius, float height, std::vector<VertexFormat> &vertices, std::vector<unsigned int> &indices, glm::vec3 color);

    float compute_rail_height(int x, int y, float desired_height, bool &modify);

    void add_rail(glm::ivec2 P0, glm::ivec2 P1, int step_size);
    void add_rail(int source_x, int source_y, int end_x, int end_y, int step_size);
    void flat_for_rail(int source_x, int source_y, int end_x, int end_y);

    void generate_request(int cnt);

    void create_grid();

    float min_h, max_h;
    int lx, ly;

    bool game_over;
    float mission_time;
    float internal_time;

    std::vector<STATION_TYPE> request;

    std::vector<station_data> station_timers;
    std::vector<STATION_TYPE> resource_types;
    std::vector<glm::ivec2> station_pos;

    std::vector<std::vector<float>> corner_values;
    std::vector<std::vector<int>> corner_power;

    std::vector<std::vector<float>> rails;
    std::vector<std::vector<float>> rail_heightmap;

    std::vector<std::vector<bool>> filter_map;
    std::vector<std::vector<float>> heightmap;
    std::vector<std::vector<glm::vec3>> normalmap;

    std::vector<RiverProperties> rivers;
    perlin::Perlin2D *river_noise;

    std::vector<std::vector<grid_cell>> grid;
};